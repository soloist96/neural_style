import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
# from matplotlib import pyplot as plt
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision import transforms

import utils
from transformer_net import TransformerNet
from vgg16 import Vgg16


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        kwargs = {}

    training_set = np.loadtxt(args.dataset, dtype=np.float32)
    training_set_size = training_set.shape[1]
    num_batch = int(training_set_size / args.batch_size)

    transformer = TransformerNet()
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16()
    utils.init_vgg16(args.vgg_model_dir)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))

    if args.cuda:
        transformer.cuda()
        vgg.cuda()

    style = np.loadtxt(args.style_image, dtype=np.float32)
    style = style.reshape((1, 1, args.style_size_x, args.style_size_y))
    style = torch.from_numpy(style)
    style = style.repeat(args.batch_size, 3, 1, 1)
    if args.cuda:
        style = style.cuda()
    style_v = Variable(style, volatile=True)
    style_v = utils.subtract_imagenet_mean_batch(style_v)
    features_style = vgg(style_v)
    gram_style = [utils.gram_matrix(y) for y in features_style]

    # Hard data
    if args.hard_data:
        hard_data = np.loadtxt(args.hard_data_file)
        # if not isinstance(hard_data[0], list):
        #     hard_data = [hard_data]


    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        # for batch_id, (x, _) in enumerate(train_loader):
        for batch_id in range(num_batch):
            x = training_set[:, batch_id * args.batch_size : (batch_id+1) * args.batch_size]
            n_batch = x.shape[1]
            count += n_batch
            x = x.transpose()
            x = x.reshape((n_batch, 1 , args.image_size_x, args.image_size_y))

            # plt.imshow(x[0,:,:,:].squeeze(0))
            # plt.show()
            x = torch.from_numpy(x).float()

            optimizer.zero_grad()

            x = Variable(x)
            if args.cuda:
                x = x.cuda()

            y = transformer(x)

            if args.hard_data:
                hard_data_loss = 0
                num_hard_data = 0
                for hd in hard_data:
                    hard_data_loss += args.hard_data_weight * (y[:, 0, hd[1], hd[0]] - hd[2]*255.0).norm()**2 / n_batch
                    num_hard_data += 1
                hard_data_loss /= num_hard_data

            y = y.repeat(1, 3, 1, 1)
            # x = Variable(utils.preprocess_batch(x))

            # xc = x.data.clone()
            # xc = xc.repeat(1, 3, 1, 1)
            # xc = Variable(xc, volatile=True)


            y = utils.subtract_imagenet_mean_batch(y)
            # xc = utils.subtract_imagenet_mean_batch(xc)

            features_y = vgg(y)
            # features_xc = vgg(xc)

            # f_xc_c = Variable(features_xc[1].data, requires_grad=False)

            # content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)

            style_loss = 0.
            for m in range(len(features_y)):
                gram_s = Variable(gram_style[m].data, requires_grad=False)
                gram_y = utils.gram_matrix(features_y[m])
                style_loss += args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])

            # total_loss = content_loss + style_loss

            total_loss = style_loss

            if args.hard_data:
                total_loss += hard_data_loss

            total_loss.backward()
            optimizer.step()

            # agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]

            if (batch_id + 1) % args.log_interval == 0:
                if args.hard_data:
                    mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\thard_data: {:.6f}\ttotal: {:.6f}".format(
                        time.ctime(), e + 1, count, num_batch,
                                      agg_content_loss / (batch_id + 1),
                                      agg_style_loss / (batch_id + 1),
                                      hard_data_loss.data[0],
                                      (agg_content_loss + agg_style_loss) / (batch_id + 1)
                    )
                else:
                    mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                        time.ctime(), e + 1, count, num_batch,
                                      agg_content_loss / (batch_id + 1),
                                      agg_style_loss / (batch_id + 1),
                                      (agg_content_loss + agg_style_loss) / (batch_id + 1)
                    )
                print(mesg)

    # save model
    transformer.eval()
    transformer.cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

'''
def stylize(args):
    # content_image = utils.tensor_load_rgbimage(args.content_image, scale=args.content_scale)
    # content_image = content_image.unsqueeze(0)
    content_image = np.loadtxt(args.content_image)
    upsample_ratio = 8
    content_image = content_image[:, :200]
    num_images = content_image.shape[1]
    content_image = content_image.transpose()
    content_image = content_image.reshape((num_images, 1 , args.image_size_x, args.image_size_y))
    content_image = torch.from_numpy(content_image).float()
    if args.cuda:
        content_image = content_image.cuda()
    content_image = Variable(content_image, volatile=True)
    style_model = TransformerNet()
    style_model.load_state_dict(torch.load(args.model))

    if args.cuda:
        style_model.cuda()

    output_model = style_model(content_image)
    # output_image = output_image.numpy()
    output_model = output_model.data
    output_image = output_model.repeat(1, 3, 1, 1)

    output_model = output_model.numpy().astype(float)
    output_model = output_model.reshape((num_images, args.image_size_x * args.image_size_y * upsample_ratio**2))
    output_model = output_model.transpose()
    np.savetxt(args.output_model, output_model)
    # output_image = output_image.reshape((args.image_size*args.image_size, num_images))
    # for k in range(num_images):
    #     utils.tensor_save_bgrimage(output_image[k], args.output_image + str(k) + '.png', args.cuda)
'''


def stylize(args):
    # content_image = utils.tensor_load_rgbimage(args.content_image, scale=args.content_scale)
    # content_image = content_image.unsqueeze(0)
    content_image = np.loadtxt(args.content_image)
    upsample_ratio = 8
    batch_size = 100
    #content_image = content_image[:, :200]
    num_images = content_image.shape[1]

    num_batch = int(content_image.shape[1] / batch_size)

    output_model_total = []
    for batch_id in range(num_batch):
        print('[{}]/[{}] iters '.format(batch_id + 1, num_batch))
        x = content_image[:, batch_id * batch_size : (batch_id+1) * batch_size]

    
        x = x.transpose()
        x = x.reshape((-1, 1 , args.image_size_x, args.image_size_y))
        x = torch.from_numpy(x).float()
        if args.cuda:
            x = x.cuda()
        x = Variable(x, volatile=True)
        style_model = TransformerNet()
        style_model.load_state_dict(torch.load(args.model))

        if args.cuda:
            style_model.cuda()

        output_model = style_model(x)
        # output_image = output_image.numpy()
        output_model = output_model.data
        output_image = output_model.repeat(1, 3, 1, 1)

        output_model = output_model.cpu().numpy().astype(float)
        output_model = output_model.reshape((batch_size, args.image_size_x * args.image_size_y * upsample_ratio**2))
        output_model = output_model.transpose()

        output_model_total.append(output_model)
    output_model_total = np.hstack(output_model_total)

    np.savetxt(args.output_model, output_model_total)
    


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train",
                                             help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--vgg-model-dir", type=str, required=True,
                                  help="directory for vgg, if model is not present in the directory it is downloaded")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--image-size-x", type=int, default=256,
                                  help="size of training images, default is 256")
    train_arg_parser.add_argument("--image-size-y", type=int, default=256,
                                  help="size of training images, default is 256")
    train_arg_parser.add_argument("--style-size-x", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--style-size-y", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1.0,
                                  help="weight for content-loss, default is 1.0")
    train_arg_parser.add_argument("--style-weight", type=float, default=5.0,
                                  help="weight for style-loss, default is 5.0")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 0.001")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--hard-data-weight", type=float, default=1.0e5,
                                  help="weight for hard data loss, default is 1.0e5")
    train_arg_parser.add_argument("--hard-data-file", type=str, default="images/style-images/",
                                  help="file that contains hard data")
    train_arg_parser.add_argument("--hard-data", type=int, default= 0,
                                  help="1 - honor to hard data, 0 - no hard data")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--image-size-x", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    eval_arg_parser.add_argument("--image-size-y", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--output-model", type=str, required=True,
                                 help="file for saving the output model")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
