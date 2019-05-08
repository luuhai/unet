import cv2
import glob

from eval_segm import *

eval_dir = "results/deep_128_leaky_aug_40e"
list_gt = glob.glob(eval_dir + "/*gt.jpg")
total_p_acc = 0
total_m_acc = 0
total_m_iu = 0

for gt in list_gt:
    gt_img = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
    final = gt[:-7] + "_final.png"
    final_img = cv2.imread(final, cv2.IMREAD_GRAYSCALE)

    p_acc = pixel_accuracy(gt_img, final_img)
    m_acc = mean_accuracy(gt_img, final_img)
    m_iu = mean_IU(gt_img, final_img)

    print("\nFile name: ", final)
    print("Pixel accuracy: ", p_acc)
    print("Mean accuracy: ", m_acc)
    print("Mean IOU: ", m_iu)

    total_p_acc += p_acc
    total_m_acc += m_acc
    total_m_iu += m_iu


print("\nAvg pixel accuracy: ", total_p_acc / len(list_gt))
print("Avg mean accuracy: ", total_m_acc / len(list_gt))
print("Avg mean IOU: ", total_m_iu / len(list_gt))
