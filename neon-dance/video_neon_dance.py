import neon_util
import cv2
import argparse
import sys
import time 

##### Khởi tạo argpaser để lấy đối số
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--input',
                    help='Path to read video',
                    required=True)

parser.add_argument('-o', '--output',
                    help='Path to write video',
                    required=True)

parser.add_argument('--time-span',
                    type=float,
                    help='After a <span> second, save neon human',
                    default=0.15)

parser.add_argument('--n-humans',
                    type=int,
                    help='Max # of neon humans',
                    default=3)

parser.add_argument('-clg', '--color-group',
                    type=int,
                    help=f'Choose color group to paint neon human. {neon_util.RED}: red, {neon_util.BLUE}: blue, {neon_util.GREEN}: green, {neon_util.RANDOM}: random',
                    default=neon_util.BLUE)

parser.add_argument('-v', '--verbose',
                    type=int,
                    help='Show process of making video',
                    default=0)

parser.add_argument('-mdc', '--min-det-conf',
                    type=float,
                    help='min detection confidence',
                    default=0.5)

parser.add_argument('-mtc', '--min-track-conf',
                    type=float,
                    help='min tracking confidence',
                    default=0.5)

##### Kiểm tra các đối số có phù hợp hay không
args = parser.parse_args()
assert args.color_group in range(len(neon_util.color_groups))
assert 0 <= args.min_det_conf <= 1
assert 0 <= args.min_track_conf <= 1

##### Xử lý video
# Khởi tạo VideoCapture để đọc video
cap = cv2.VideoCapture(args.input)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = int(cap.get(cv2.CAP_PROP_FPS))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f'size: {size}')
print(f'fps: {fps}')
print(f'n_frames: {n_frames}')

# Khởi tạo NeonHumanSaver để lưu trữ người neon và tạo ảnh cuối hoàn chỉnh
neon_human_saver = neon_util.NeonHumanSaver(size[0], size[1],
                                       n_humans=args.n_humans,
                                       min_detection_confidence=args.min_det_conf,
                                       min_tracking_confidence=args.min_track_conf,
                                       time_span=args.time_span, fps=fps,
                                       color_group=args.color_group)

# Khởi tạo VideoWriter để ghi ảnh vào video
writer = cv2.VideoWriter(args.output, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         fps, size)

# Xử lý video
start = time.time()
cnt = 0
while cap.isOpened():
    if cnt % 100 == 0:
        print(f'{cnt}/{n_frames}')
    cnt += 1

    ret, frame = cap.read()
    if ret == False:
        break

    # Tạo ảnh cuối hoàn chỉnh
    frame = neon_human_saver.draw_neon_humans(frame)

    # Ghi vào video và show quá trình lên webcam nếu verbose != 0
    writer.write(frame)
    if args.verbose:
        cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

total = time.time() - start
print(f'total time: {total}, fps: {n_frames / total}')
cap.release()
writer.release()
cv2.destroyAllWindows()
