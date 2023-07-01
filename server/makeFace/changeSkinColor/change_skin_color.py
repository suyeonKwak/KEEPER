import cv2
import numpy as np
from makeFace.changeSkinColor.get_skin_color import get_skin_color

# from get_skin_color import get_skin_color


def change_skin_color(target_path, virtual_path):
    target_img = cv2.imread(target_path)
    virtual_img = cv2.imread(virtual_path)
    hsv_image = cv2.cvtColor(virtual_img, cv2.COLOR_BGR2HSV)

    # 피부색 영역 추출을 위한 범위 설정
    lower_skin_color = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin_color = np.array([20, 255, 255], dtype=np.uint8)

    # # 피부색 영역을 마스크로 추출
    skin_mask = cv2.inRange(hsv_image, lower_skin_color, upper_skin_color)

    _, target_hsv, _ = get_skin_color(target_img)
    # _, mean_hsv, skin_mask = get_skin_color(virtual_img)
    _, mean_hsv, _ = get_skin_color(virtual_img)
    diff_hsv = target_hsv - mean_hsv

    # # 피부색 영역에 대해 평균 HSV 값을 적용
    # hsv_image[skin_mask.squeeze() > 0] += diff_hsv.astype(np.uint8)
    hsv_image[skin_mask > 0] += diff_hsv.astype(np.uint8)
    # hsv_image[skin_mask > 0] = [120, 75, 230]

    # 변경된 HSV 이미지를 BGR로 변환
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return result_image


if __name__ == "__main__":
    # target_img = cv2.imread("C:/Users/user/Desktop/Coders/makeFace/data/Target.png")
    # virtual_img = cv2.imread(
    #     "C:/Users/user/Desktop/Coders/makeFace/generated/test_v3.png"
    # )
    # changed_img = change_skin_color(target_img, virtual_img)
    # cv2.imshow("change skin color", changed_img)
    # cv2.imwrite(
    #     "C:/Users/user/Desktop/Coders/makeFace/generated/changed_test.png", changed_img
    # )
    # cv2.waitKey(0)

    target_img = cv2.imread(
        "/content/drive/MyDrive/coders/server_yoojin/makeFace/data/target2.png"
    )
    virtual_img = cv2.imread(
        "/content/drive/MyDrive/coders/server_yoojin/makeFace/generated/v0.png"
    )
    changed_img = change_skin_color(target_img, virtual_img)
    cv2.imshow("change skin color", changed_img)
    cv2.imwrite(
        "/content/drive/MyDrive/coders/server_yoojin/makeFace/generated/changed.png",
        changed_img,
    )
    cv2.waitKey(0)
