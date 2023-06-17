import neon_util
import cv2
import argparse

##### Initialize argparser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-s', '--save',
                    type=int,
                    help='Save video',
                    default=0)

parser.add_argument('--n_humans',
                    type=int,
                    help='Max # of neon humans',
                    default=3)

parser.add_argument('-mdc', '--min-det-conf',
                    type=float,
                    help='min detection confidence',
                    default=0.5)

parser.add_argument('-mtc', '--min-track-conf',
                    type=float,
                    help='min tracking confidence',
                    default=0.5)

parser.add_argument('--span',
                    type=float,
                    help='After a <span> frames, save neon human',
                    default=2)

parser.add_argument('-clg', '--color-group',
                    type=int,
                    help=f'Choose color group to paint neon human. {neon_util.WARM}: Warm, {neon_util.COOL}: Cool, {neon_util.BOTH}: BOTH',
                    default=2)

##### Check argument
args = parser.parse_args()
assert args.color_group in (neon_util.WARM, neon_util.COOL, neon_util.BOTH)
assert 0 <= args.min_det_conf <= 1
assert 0 <= args.min_track_conf <= 1

##### Process video
# Initialize VideoCapture object
cap = cv2.VideoCapture(0)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Initialize NeonHumanSaver object to save and draw neon human
neon_human_saver = neon_util.NeonHumanSaver(size[0], size[1],
                                       n_humans=args.n_humans,
                                       min_detection_confidence=args.min_det_conf,
                                       min_tracking_confidence=args.min_track_conf,
                                       span=1, faded=args.faded,
                                       color_group=args.color_group)

# Initializa VideoWriter object to write video
if args.save:
    writer = cv2.VideoWriter('output.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         1, size)


cnt = 0
while cap.isOpened():
    if cnt % 100 == 0:
        print(f'{cnt} (frames)')
    cnt += 1

    ret, frame = cap.read()
    if ret == False:
        break

    # Draw neon humans
    frame = neon_human_saver.draw_neon_humans(frame)
    frame = cv2.flip(frame, 1)

    # Write and show frame
    if args.save:
        writer.write(frame)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
if args.save:
    writer.release()
cv2.destroyAllWindows()