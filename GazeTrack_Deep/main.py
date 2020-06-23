mport argparse
import os
import queue
import threading
import time
from datasources import Video, Webcam
import coloredlogs
import cv2 as cv
import numpy as np
import tensorflow as tf
import util.gaze
import nnarch

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description='Demonstration of landmarks localization.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('--from_video', type=str, help='Use this video path instead of webcam')
    parser.add_argument('--record_video', type=str, help='Output path of video of demonstration.')
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--headless', action='store_true')

    parser.add_argument('--fps', type=int, default=60, help='Desired sampling rate of webcam')
    parser.add_argument('--camera_id', type=int, default=0, help='ID of webcam to use')

    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session(config=session_config) as session:

        batch_size = 2
        if args.from_video:
            assert os.path.isfile(args.from_video)
            data_source = Video(args.from_video,
                                tensorflow_session=session, batch_size=batch_size,
                                data_format='NCHW' if gpu_available else 'NHWC',
                                eye_image_shape=(108, 180))
        else:
            data_source = Webcam(tensorflow_session=session, batch_size=batch_size,
                                 camera_id=args.camera_id, fps=args.fps,
                                 data_format='NCHW' if gpu_available else 'NHWC',
                                 eye_image_shape=(36, 60))

        if args.from_video:
            model = nnarch(
                session, train_data={'videostream': data_source},
                first_layer_stride=3,
                num_modules=3,
                num_feature_maps=64,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    },
                ],
            )
         else:
            model = nnarch(
                session, train_data={'videostream': data_source},
                first_layer_stride=1,
                num_modules=2,
                num_feature_maps=32,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    },
                ],
            )
            
          if args.record_video:
              video_out = None
              video_out_queue = queue.Queue()
              video_out_should_stop = False
              video_out_done = threading.Condition()

              def _record_frame():
                  global video_out
                  last_frame_time = None
                  out_fps = 30
                  out_frame_interval = 1.0 / out_fps
                  while not video_out_should_stop:
                      frame_index = video_out_queue.get()
                      if frame_index is None:
                          break
                      assert frame_index in data_source._frames
                      frame = data_source._frames[frame_index]['bgr']
                      h, w, _ = frame.shape
                      if video_out is None:
                          video_out = cv.VideoWriter(
                              args.record_video, cv.VideoWriter_fourcc(*'H264'),
                              out_fps, (w, h),
                          )
                      now_time = time.time()
                      if last_frame_time is not None:
                          time_diff = now_time - last_frame_time
                          while time_diff > 0.0:
                              video_out.write(frame)
                              time_diff -= out_frame_interval
                      last_frame_time = now_time
                  video_out.release()
                  with video_out_done:
                      video_out_done.notify_all()
              record_thread = threading.Thread(target=_record_frame, name='record')
              record_thread.daemon = True
              record_thread.start()
              
         if args.record_video and video_out is not None:
            video_out_should_stop = True
            video_out_queue.put_nowait(None)
            with video_out_done:
                video_out_done.wait()
            
                
