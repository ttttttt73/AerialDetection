import cv2
import numpy as np
import os
import fnmatch


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


def extend(path, extend_div_scale):
    img_dir = path.rsplit("/", 1)[0]
    img_name = path.rsplit("/", 1)[1]
    # print(img_name)
    
    tile = img_name.split("_", 2)
    tile = [tile[0], tile[1]]
    tile_x = int(tile[0])
    tile_y = int(tile[1])
    surround_coords = [[tile_x-1, tile_y-1], [tile_x, tile_y-1], [tile_x+1, tile_y-1], 
                    [tile_x-1, tile_y],         None,              [tile_x+1, tile_y], 
                    [tile_x-1, tile_y+1], [tile_x, tile_y+1], [tile_x+1, tile_y+1]]
    
    img = cv2.imread(path)
    img_h, img_w= img.shape[:2]
    # print("img.shape: {}, img_w: {}, img_h: {}".format(img.shape, img_w, img_h))
    
    # print("| {} | {} | {} |".format(surround_coords[0], surround_coords[1], surround_coords[2]))
    # print("| {} | {} | {} |".format(surround_coords[3], surround_coords[4], surround_coords[5]))
    # print("| {} | {} | {} |".format(surround_coords[6], surround_coords[7], surround_coords[8]))

    surround_img_names = [None] * 9
    for i, coord in enumerate(surround_coords):
        if coord is not None:
            surround_img_name_filter = "{}_{}_".format(coord[0], coord[1])
            for f in os.listdir(img_dir):
                if fnmatch.fnmatch(f, surround_img_name_filter+"*.png"):
                    # print(f)
                    surround_img_names[i] = f
                

    # print("| {} | {} | {} |".format(surround_img_names[0], surround_img_names[1], surround_img_names[2]))
    # print("| {} | {} | {} |".format(surround_img_names[3], surround_img_names[4], surround_img_names[5]))
    # print("| {} | {} | {} |".format(surround_img_names[6], surround_img_names[7], surround_img_names[8]))

    surround_imgs = [None] * 9
    for i, name in enumerate(surround_img_names):
        if name is not None:
            surround_img = cv2.imread(os.path.join(img_dir, name))
            surround_imgs[i] = surround_img
            '''cv2.imshow("surround_imgs[{}]".format(i), surround_imgs[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
        else:
            surround_imgs[i] = np.zeros(shape=(img_h, img_w, 3), dtype=np.uint8)
    
    surround_imgs[4] = img
    
    # res = np.zeros(shape=(img_h*2, img_w*2, 3), dtype=np.uint8)
    # res[int(img_h/2):int(img_h/2)+img_h, int(img_w/2):int(img_w/2)+img_w, :3] = img
    # surround_imgs[0]

    res = concat_tile([[surround_imgs[0], surround_imgs[1], surround_imgs[2]], 
                        [surround_imgs[3], surround_imgs[4], surround_imgs[5]], 
                        [surround_imgs[6], surround_imgs[7], surround_imgs[8]]])
    
    scaled_img_h = extend_div_scale * img_h
    scaled_img_w = extend_div_scale * img_w

    scaeld_start_h = (1 - extend_div_scale) * img_h
    scaeld_start_w = (1 - extend_div_scale) * img_w

    res = res[int(scaeld_start_h):int(scaeld_start_h+(scaled_img_h*2)+img_h), int(scaeld_start_w):int(scaeld_start_w)+int((scaled_img_w*2)+img_w)]
    # res = res[int(img_h/2):int(img_h/2)+int(img_h*2), int(img_w/2):int(img_w/2)+int(img_w*2)]

    # print("res.shape: ", res.shape)
    '''cv2.circle(img, (540, 1), radius=1, color=(0, 0, 255),
                             thickness=5)
    cv2.circle(img, (-30, -30), radius=1, color=(0, 0, 255),
                             thickness=5)
    cv2.circle(img, (1000, 1000), radius=1, color=(0, 0, 255),
                             thickness=5)'''
    
    '''cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    return res

    

if __name__ == "__main__":
    path = "./area2_10/MO-11111111110/0_0_35.69491915900303_139.69317728042603_35.69520833_139.6928125_35.69354167_139.6971875.png"
    extend_div_scale = 1/4
    extend(path, extend_div_scale)
