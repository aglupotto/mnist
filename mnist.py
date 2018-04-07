from matplotlib import pyplot
import numpy as np
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe


from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew

from caffe2.proto import caffe2_pb2

# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=-1'])


# This section preps your image and test set in a lmdb database
def DownloadResource(url, path):
    '''Downloads resources from s3 by url and unzips them to the provided path'''
    import requests, zipfile, StringIO
    print("Downloading... {} to {}".format(url, path))
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall(path)
    print("Completed download and extraction.")


def set_db_paths():

    current_folder = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks')
    data_folder = os.path.join(current_folder,'tutorial_data','mnist')
    root_folder = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')
    db_missing = True


    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print("Data folder not found. Generated at: {}".format(data_folder))

    if os.path.exists(os.path.join(data_folder,"mnist-train-nchw-lmdb")):
        print("lmdb train db found")
    else:
        db_missing = True

    #attempt the download of the db if either was missing
    if db_missing:
        db_url = "http://download.caffe2.ai/databases/mnist-lmdb.zip"
        try:
            DownloadResource(db_url, data_folder)
        except Exception as ex:
            print("Failed to download dataset. Please download it manually from {}".format(db_url))
            print("Unzip it and place the two database folders here: {}".format(data_folder))
            raise ex

    if os.path.exists(root_folder):
        print("Looks like you ran this before, so we need to cleanup those old files...")
        shutil.rmtree(root_folder)

    os.makedirs(root_folder)
    workspace.ResetWorkspace(root_folder)

    print("training data folder:" + data_folder)
    print("workspace root folder:" + root_folder)

#load data. return data w/ shape (bach_size, n_ch, width, height)
def AddInput(model,batch_size,db,db_type):
    #load the data
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)

    #cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)

    print(data)
    #scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    print(data)

    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)

    return data,label

def AddLeNetModel(model, data):
    '''
        This part is the standard LeNet model: from data to the softmax prediction.

        For each convolutional layer we specify dim_in - number of input channels
        and dim_out - number or output channels. Also each Conv and MaxPool layer changes the
        image size. For example, kernel of size 5 reduces each side of an image by 4.

        While when we have kernel and stride sizes equal 2 in a MaxPool layer, it divides
        each side in half.
        '''
    # Image size: 28 x 28 -> 24 x 24
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=100, kernel=5)
    # Image size: 8 x 8 -> 4 x 4
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=100 * 4 * 4, dim_out=500)

    relu = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, relu, 'pred', 500, 10)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax

def AddAccuracy(model, softmax, label):
    """Adds an accuracy op to the model"""
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy

def AddTrainingOperators(model, softmax, label):
    """Adds training operators to the model."""
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    AddAccuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = brew.iter(model, "iter")
    # set the learning rate schedule
    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - ModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)
def AddBookkeepingOperators(model):
    """This adds a few bookkeeping operators that we can inspect later.

    These operators do not affect the training procedure: they only collect
    statistics and prints them to file or to logs.
    """
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
    # Now, if we really want to be verbose, we can summarize EVERY blob
    # that the model produces; it is probably not a good idea, because that
    # is going to take time - summarization do not come for free. For this
    # demo, we will only show how to summarize the parameters and their
    # gradients.

if __name__ == '__main__':
    #download db and set paths
    #set_db_paths()
    current_folder = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks')
    data_folder = os.path.join(current_folder, 'tutorial_data', 'mnist')
    root_folder = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')


    #load data
    arg_scope = {"order": "NCHW"}
    train_model = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope)

    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
        data, label = AddInput(
        train_model, batch_size=64,
        db=os.path.join(data_folder, 'mnist-train-nchw-lmdb'),
        db_type='lmdb')

        softmax = AddLeNetModel(train_model, data)
        AddTrainingOperators(train_model, softmax, label)
        AddBookkeepingOperators(train_model)

    # Testing model. We will set the batch size to 100, so that the testing
    # pass is 100 iterations (10,000 images in total).
    # For the testing model, we need the data input part, the main LeNetModel
    # part, and an accuracy part. Note that init_params is set False because
    # we will be using the parameters obtained from the train model.

    test_model = model_helper.ModelHelper(
        name="mnist_test", arg_scope=arg_scope, init_params=False)
    data, label = AddInput(
        test_model, batch_size=100,
        db=os.path.join(data_folder, 'mnist-test-nchw-lmdb'),
        db_type='lmdb')
    softmax = AddLeNetModel(test_model, data)
    AddAccuracy(test_model, softmax, label)

    # Deployment model. We simply need the main LeNetModel part.
    deploy_model = model_helper.ModelHelper(
        name="mnist_deploy", arg_scope=arg_scope, init_params=False)
    AddLeNetModel(deploy_model, "data")
    # You may wonder what happens with the param_init_net part of the deploy_model.
    # No, we will not use them, since during deployment time we will not randomly
    # initialize the parameters, but load the parameters from the db.

    with open(os.path.join(root_folder, "train_net.pbtxt"), 'w') as fid:
        fid.write(str(train_model.net.Proto()))
    with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fid:
        fid.write(str(train_model.param_init_net.Proto()))
    with open(os.path.join(root_folder, "test_net.pbtxt"), 'w') as fid:
        fid.write(str(test_model.net.Proto()))
    with open(os.path.join(root_folder, "test_init_net.pbtxt"), 'w') as fid:
        fid.write(str(test_model.param_init_net.Proto()))
    with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
        fid.write(str(deploy_model.net.Proto()))
    print("Protocol buffers files have been created in your root folder: " + root_folder)


    #star running everything
    train_model.net.RunAllOnGPU(gpu_id=0, use_cudnn=True)
    train_model.param_init_net.RunAllOnGPU(gpu_id=0, use_cudnn=True)


    ################DON'T WORK HEREE###############
    #initialize the network
    workspace.RunNetOnce(train_model.param_init_net)

    #protobuf into workspace
    workspace.CreateNet(train_model.net)

    #set numpy array to save accuracies
    total_iters = 200
    accuracy = np.zeros(total_iters)
    loss = np.zeros(total_iters)

    #device = core.DeviceOption(caffe2_pb2.CUDA, 0)
    #loop 200
    for i in range(total_iters):
        workspace.RunNet(train_model.net.Proto().name)
        accuracy[i] = workspace.FetchBlob('accuracy')
        loss[i] = workspace.FetchBlob('loss')

    # After the execution is done, let's plot the values.
    pyplot.plot(loss, 'b')
    pyplot.plot(accuracy, 'r')
    pyplot.legend(('Loss', 'Accuracy'), loc='upper right')