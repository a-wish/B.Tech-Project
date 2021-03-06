import argparse

import coloredlogs
import tensorflow as tf

if __name__ == '__main__':

   
    parser = argparse.ArgumentParser(description='Train the Deep Pictorial Gaze model.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    for i in range(0, 15):
       
        person_id = 'p%02d' % i
        other_person_ids = ['p%02d' % j for j in range(15) if i != j]

        
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.ERROR)
        config=tf.ConfigProto(log_device_placement=True)
        with tf.Session(config=config) as session:
            batch_size = 32
            from datasources import HDF5Source
            from models import DPG
            model = DPG(
                session,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {
                            'combined_loss': ['hourglass', 'densenet'],
                        },
                        'metrics': ['gaze_mse', 'gaze_ang'],
                        'learning_rate': 0.0002,
                    },
                ],
                extra_tags=[person_id],

                train_data={
                    'mpi': HDF5Source(
                        session,
                        data_format='NCHW',
                        batch_size=batch_size,
                        keys_to_use=['train/' + s for s in other_person_ids],
                        hdf_path='../datasets/MPIIGaze.h5',
                        eye_image_shape=(90, 150),
                        testing=False,
                        min_after_dequeue=30000,
                        staging=True,
                        shuffle=True,
                    ),
                },
                test_data={
                    'mpi': HDF5Source(
                        session,
                        data_format='NCHW',
                        batch_size=batch_size,
                        keys_to_use=['test/' + person_id],
                        hdf_path='../datasets/MPIIGaze.h5',
                        eye_image_shape=(90, 150),
                        testing=True,
                    ),
                },
            )
            model.train(
                num_epochs=20,
            )

            model.__del__()
            session.close()
            del session
