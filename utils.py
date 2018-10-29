import cv2
import os

def resize_square(srcpath, dstpath, size=512):
    for image in [files for files in os.listdir(srcpath)]:
        img = cv2.imread(srcpath+"/"+image)

        assert img.shape[0] == img.shape[1]

        print("Image: {}\t {} -> {}".format(image, img.shape[0], size))

        resized = cv2.resize(img, (size, size))
        cv2.imwrite(dstpath + '/' + image, resized)
    print("done!")



def crop_to_square(src_path, dst_path, square=None):

    cntr = 0
    ls = [files for files in os.listdir(src_path)]

    for image in ls:
        if image.endswith('.jpg'):
            img = cv2.imread(os.path.join(src_path, image))
            h, w, _ = img.shape
            size = min(h, w)
            off_h = (h - size) // 2
            off_w = (w - size) // 2

            crop = img[off_h : off_h + size, off_w : off_w + size]

            if square:
                crop = cv2.resize(crop, (square, square))


            assert crop.shape[0] == crop.shape[1]

            cv2.imwrite(os.path.join(dst_path, "crop-"+image), crop)
            cntr += 1

        else:
            print("{} is not a proper image.".format(image))
    
    print("{} / {} images cropped {}".format(cntr, len(ls), "and resized." if square else "."))



if __name__ == "__main__":
    # resize_square("data/rawcropped2048", "data/rawcropped224", size=224)
    crop_to_square("../datasets/apple_pie/", "data/apple_pie/", 224)    