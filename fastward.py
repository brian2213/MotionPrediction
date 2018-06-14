import json
import pdb
import pickle
import cv2

import matplotlib.pyplot as plt

data_file_name = json.load(open("./Dataset/data_file_name.json", 'r'))
data = pickle.load(open("./Dataset/test-32data.save", 'rb'))
# pdb.set_trace()
image_name_sorted = list(data_file_name)
image_name_sorted.sort()

left_hand = []
right_hand = []

for hand in image_name_sorted:
    if hand[-1] == 'R':
        right_hand.append(hand)
    else:
        left_hand.append(hand)

get_image_index = {}
for i in range(len(data_file_name)):
    get_image_index[data_file_name[i]] = i

left_pre = left_hand[0]
right_pre = right_hand[0]
for i in range(1, len(left_hand)):
    if i < len(left_hand) and left_hand[i][:3] == left_pre[:3]:
        pre_idx = get_image_index[left_pre]
        idx = get_image_index[left_hand[i]]
        data['label'][pre_idx] = data['label'][idx]
    if i < len(right_hand) and right_hand[i][:3] == right_pre[:3]:
        pre_idx = get_image_index[right_pre]
        idx = get_image_index[right_hand[i]]
        data['label'][pre_idx] = data['label'][idx]

    if i < len(left_hand):
        left_pre = left_hand[i]
    if i < len(right_hand):
        right_pre = right_hand[i]

idx = 4567
# plt.imshow(cv2.cvtColor(data['data'][idx], cv2.COLOR_BGR2RGB))
# plt.scatter(data['label'][idx][:, 1], data['label'][idx][:, 0])
# plt.show()
# pdb.set_trace()
data = pickle.dump(data, open("./Dataset/test-32data-fastward.save", 'wb'))
