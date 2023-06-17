import cv2
import mediapipe as mp
import numpy as np
import random

RED = 0
BLUE = 1
GREEN = 2
RANDOM = 3

color_groups = {
    RED: np.array([[0, 0, 255],
                   [52, 0, 196],
                   [121, 0, 228],
                   [167, 0, 228],
                   [0, 111, 255],
                   [0, 171, 255]]),

    BLUE: np.array([[255, 0, 171],
                    [255, 0, 102],
                    [255, 0, 43],
                    [255, 85, 0]]),

    GREEN: np.array([[94, 255, 0],
                     [0, 255, 0],
                     [0, 255, 128]])
}

# Class GlowHelper giúp tạo 4 layers cho việc tạo hiệu ứng neon, 4 layers đó là: outline, edge, blur, base
class GlowHelper():
    def create_layers(self, seg_mask):
        # Tạo ảnh eroded
        mask = seg_mask * 1.0
        eroded = cv2.erode(mask, np.ones((15, 15), np.uint8), 1)
        
        # Tạo outline
        outline = mask - eroded

        # Tạo cạnh cho outline
        edge = cv2.dilate(outline, np.ones((7, 7)), 1) 

        # Tạo blur
        blur = cv2.dilate(edge, np.ones((25, 25)), 1)
        
        # Tạo nền
        base = cv2.dilate(blur, np.ones((15, 15)), 1)

        return (outline, edge, blur, base)

# Class NeonHumanSaver giúp lưu trư những người neon, và tạo ảnh cuối hoàn chỉnh
class NeonHumanSaver():
    def __init__(self, width, height, n_humans, 
                 min_detection_confidence, 
                 min_tracking_confidence, 
                 time_span, fps, color_group):

        # Khởi tạo detector để segment mask của người trong ảnh
        self.detector = mp.solutions.pose.Pose(min_detection_confidence=min_detection_confidence,
                                               min_tracking_confidence=min_tracking_confidence,
                                               enable_segmentation=True)
        
        # Khởi tạo glow_helper để vẽ 4 layers giúp tạo hiệu ứng neon
        self.glow_helper = GlowHelper()

        self.n_humans = n_humans # Số người neon tối đa
        self.span = int(time_span * fps) # Khoảng cách tính bằng frame giữa các chuyển động của người thật và các người neon
        self.width = width # Chiều rộng của video
        self.height = height # Chiều cao của video

        self.queue = [None] * self.span * n_humans

        # Tạo màu cho các người neon
        if color_group == RANDOM:
            self.colors = [self.get_random_color() for i in range(n_humans)]
        else:
            self.colors = [color_groups[color_group][i % len(color_groups[color_group])] for i in range(n_humans)]

    def get_random_color(self):
        clg = random.randint(0, len(color_groups) - 1)
        return color_groups[clg][random.randint(0, len(color_groups[clg]) - 1)]

    def draw_neon_humans(self, cur_img):
        # Copy cur_img and assign to bg
        bg = cur_img.copy()

        # Tạo ảnh bases chung cho các người neon
        bases = np.zeros((self.height, self.width), np.bool)

        # Duyệt i từ n_humans - 1 đến 0, nếu vị trí (n_humans - 1 - i) x span
        for i in range(self.n_humans - 1, -1, -1):
            if self.queue[self.span * (self.n_humans - 1 - i)]:
                outline, edge, blur, base = self.queue[self.span * (self.n_humans - 1 - i)]

                # Blend bg với blur
                bg[blur != 0] = (bg[blur != 0] * 0.8 + (blur[:, :, None] * self.colors[i])[blur != 0] * 0.2).astype(np.uint8)

                # Ghi đè edge lên background
                bg[edge != 0] = self.colors[i]

                # Ghi đè outline lên background
                bg[outline != 0] = np.array((223, 223, 223))

                # Cập nhật bases
                bases[base != 0] = True

        # Áp dụng gaussian blur cho background
        bg = cv2.GaussianBlur(bg, (5, 5), 4)

        # Lấy mask của người trong ảnh hiện tại
        seg_mask = self.detector.process(cur_img).segmentation_mask
        if seg_mask is None:
            return cur_img
        seg_mask = (seg_mask > 0.5)

        # Tạo những thông tin mới (outline, edge, blur, base) cho ảnh hiện tại và đẩy vào hàng đợi
        self.queue.pop(0)
        self.queue.append(self.glow_helper.create_layers(seg_mask))
        
        # Ghi đè người trong ảnh hiện tại vào background
        bg[seg_mask] = cur_img[seg_mask]

        # Ghi đè vùng base trong ảnh background vào ảnh hiện tại để tạo ảnh cuối hoàn chỉnh
        cur_img[bases] = bg[bases]

        return cur_img
        