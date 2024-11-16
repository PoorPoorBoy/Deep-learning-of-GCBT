import os
import json
import numpy
import torch
from PIL import Image
from torchvision import transforms
from model import GoogLeNet



from torchvision import datasets, models, transforms

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         #transforms.CenterCrop(100),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])




    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = "D:/wzh_data2/3class_new/test/dongmai"

    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    # read class_indict
    json_path = './class_indices2.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model

    model = GoogLeNet(num_classes=3, aux_logits=False).to(device)


    # load model weights
    weights_path = "D:/wzh_data2/3class_new/googlenet_3model-23.pt"


    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    #model.load_state_dict(torch.load(weights_path, map_location=device))
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device),
                                                          strict=False)  # strict=False表示不进行精准匹配

    # prediction
    model.eval()
    batch_size = 1  # 每次预测时将多少张图片打包成一个batch
    classes_list = []
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)



            classes_list.append(int(classes))



            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))
    numpy.savetxt('predict.txt', classes_list)

    predict_0 = 0
    predict_1 = 0
    predict_2 = 0
    predict_3 = 0
    for preditc_value in classes_list:
        if preditc_value == 0:
            predict_0 += 1
        elif preditc_value == 1:
            predict_1 += 1
        elif preditc_value == 2:
            predict_2 += 1
        elif preditc_value == 3:
            predict_3 += 1

    print('0:', predict_0, '1:', predict_1, '2:', predict_2, '3:', predict_3)

if __name__ == '__main__':
    main()
