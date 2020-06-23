from collections import OrderedDict
import multiprocessing
import queue
import threading
import time

import numpy as np
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)


class BaseDataSource(object):
    

    def __init__(self,
                 tensorflow_session: tf.Session,
                 data_format: str = 'NHWC',
                 batch_size: int = 32,
                 num_threads: int = max(4, multiprocessing.cpu_count()),
                 min_after_dequeue: int = 1000,
                 fread_queue_capacity: int = 0,
                 preprocess_queue_capacity: int = 0,
                 staging=False,
                 shuffle=None,
                 testing=False,
                 ):
        
        assert tensorflow_session is not None and isinstance(tensorflow_session, tf.Session)
        assert isinstance(batch_size, int) and batch_size > 0
        if shuffle is None:
            shuffle = staging
        self.testing = testing
        if testing:
            assert not shuffle and not staging
            
        self.staging = staging
        self.shuffle = shuffle
        self.data_format = data_format.upper()
        assert self.data_format == 'NHWC' or self.data_format == 'NCHW'
        self.batch_size = batch_size
        self.num_threads = num_threads
        self._tensorflow_session = tensorflow_session
        self._coordinator = tf.train.Coordinator()
        self.all_threads = []

        
        self._fread_queue_capacity = fread_queue_capacity
        if self._fread_queue_capacity == 0:
            self._fread_queue_capacity = (num_threads + 1) * batch_size
        self._fread_queue = queue.Queue(maxsize=self._fread_queue_capacity)

        with tf.variable_scope(''.join(c for c in self.short_name if c.isalnum())):
            
            labels, dtypes, shapes = self._determine_dtypes_and_shapes()
            self._preprocess_queue_capacity = (min_after_dequeue + (num_threads + 1) * batch_size
                                               if preprocess_queue_capacity == 0
                                               else preprocess_queue_capacity)
            if shuffle:
                self._preprocess_queue = tf.RandomShuffleQueue(
                        capacity=self._preprocess_queue_capacity,
                        min_after_dequeue=min_after_dequeue,
                        dtypes=dtypes, shapes=shapes,
                )
            else:
                self._preprocess_queue = tf.FIFOQueue(
                        capacity=self._preprocess_queue_capacity,
                        dtypes=dtypes, shapes=shapes,
                )
            self._tensors_to_enqueue = OrderedDict([
                (label, tf.placeholder(dtype, shape=shape, name=label))
                for label, dtype, shape in zip(labels, dtypes, shapes)
            ])

            self._enqueue_op = \
                self._preprocess_queue.enqueue(tuple(self._tensors_to_enqueue.values()))
            self._preprocess_queue_close_op = \
                self._preprocess_queue.close(cancel_pending_enqueues=True)
            self._preprocess_queue_size_op = self._preprocess_queue.size()
            self._preprocess_queue_clear_op = \
                self._preprocess_queue.dequeue_up_to(self._preprocess_queue.size())
            if not staging:
                output_tensors = self._preprocess_queue.dequeue_many(self.batch_size)
                if not isinstance(output_tensors, list):
                    output_tensors = [output_tensors]
                self._output_tensors = dict([
                    (label, tensor) for label, tensor in zip(labels, output_tensors)
                ])
            else:
                
                self._staging_area = tf.contrib.staging.StagingArea(
                    dtypes=dtypes,
                    shapes=[tuple([batch_size] + list(shape)) for shape in shapes],
                    capacity=1,  # This does not have to be high
                )
                self._staging_area_put_op = \
                    self._staging_area.put(self._preprocess_queue.dequeue_many(batch_size))
                self._staging_area_clear_op = self._staging_area.clear()

                self._output_tensors = dict([
                    (label, tensor) for label, tensor in zip(labels, self._staging_area.get())
                ])

        logger.info('Initialized data source: "%s"' % self.short_name)

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        
        if self.__cleaned_up:
            return

        
        fread_threads = [t for t in self.all_threads if t.name.startswith('fread_')]
        preprocess_threads = [t for t in self.all_threads if t.name.startswith('preprocess_')]
        transfer_threads = [t for t in self.all_threads if t.name.startswith('transfer_')]

        self._coordinator.request_stop()

       
        while True:
            try:
                self._fread_queue.get_nowait()
            except queue.Empty:
                break
            time.sleep(0.1)

        
        for _ in range(self.batch_size * self.num_threads):
            self._fread_queue.put(None)
        self._tensorflow_session.run(self._preprocess_queue_close_op)
        if self.staging:
            self._tensorflow_session.run(self._staging_area_clear_op)

        self._coordinator.join(self.all_threads, stop_grace_period_secs=5)
        self.__cleaned_up = True

    def reset(self):
        
        assert self.testing is True

        
        self._coordinator.request_stop()
        with self._fread_queue.mutex:  
            self._fread_queue.queue.clear()
        for _ in range(2*self.num_threads):
            self._fread_queue.put(None)
        while True:  
            preprocess_queue_size = self._tensorflow_session.run(self._preprocess_queue_size_op)
            if preprocess_queue_size == 0:
                break
            self._tensorflow_session.run(self._preprocess_queue_clear_op)
            time.sleep(0.1)
        while True:  
            try:
                self._fread_queue.get_nowait()
            except queue.Empty:
                break
            time.sleep(0.1)
        self._coordinator.join(self.all_threads, stop_grace_period_secs=5)

        
        self._coordinator.clear_stop()
        self.create_and_start_threads()

    def _determine_dtypes_and_shapes(self):
        
        while True:
            raw_entry = next(self.entry_generator(yield_just_one=True))
            if raw_entry is None:
                continue
            preprocessed_entry_dict = self.preprocess_entry(raw_entry)
            if preprocessed_entry_dict is not None:
                break
        labels, values = zip(*list(preprocessed_entry_dict.items()))
        dtypes = [value.dtype for value in values]
        shapes = [value.shape for value in values]
        return labels, dtypes, shapes

    def entry_generator(self, yield_just_one=False):
        
        raise NotImplementedError('BaseDataSource::entry_generator not implemented.')

    def preprocess_entry(self, entry):
        
        raise NotImplementedError('BaseDataSource::preprocess_entry not implemented.')

    def read_entry_job(self):
        
        read_entry = self.entry_generator()
        while not self._coordinator.should_stop():
            try:
                entry = next(read_entry)
            except StopIteration:
                if not self.testing:
                    continue
                else:
                    logger.debug('Reached EOF in %s' % threading.current_thread().name)
                    break
            if entry is not None:
                self._fread_queue.put(entry)
        read_entry.close()
        logger.debug('Exiting thread %s' % threading.current_thread().name)

    def preprocess_job(self):
        
        while not self._coordinator.should_stop():
            raw_entry = self._fread_queue.get()
            if raw_entry is None:
                return
            preprocessed_entry_dict = self.preprocess_entry(raw_entry)
            if preprocessed_entry_dict is not None:
                feed_dict = dict([(self._tensors_to_enqueue[label], value)
                                  for label, value in preprocessed_entry_dict.items()])
                try:
                    self._tensorflow_session.run(self._enqueue_op, feed_dict=feed_dict)
                except (tf.errors.CancelledError, RuntimeError):
                    break
        logger.debug('Exiting thread %s' % threading.current_thread().name)

    def transfer_to_job(self):
        
        while not self._coordinator.should_stop():
            try:
                self._tensorflow_session.run(self._staging_area_put_op)
            except tf.errors.CancelledError or tf.errors.OutOfRangeError:
                break
        logger.debug('Exiting thread %s' % threading.current_thread().name)

    def create_threads(self):
        
        name = self.short_name
        self.all_threads = []

        def _create_and_register_thread(*args, **kwargs):
            thread = threading.Thread(*args, **kwargs)
            thread.daemon = True
            self.all_threads.append(thread)

        for i in range(self.num_threads):
            
            _create_and_register_thread(target=self.read_entry_job, name='fread_%s_%d' % (name, i))

            
            _create_and_register_thread(target=self.preprocess_job,
                                        name='preprocess_%s_%d' % (name, i))

        if self.staging:
            
            _create_and_register_thread(target=self.transfer_to_job,
                                        name='transfer_%s_%d' % (name, i))

    def start_threads(self):
        
        assert len(self.all_threads) > 0
        for thread in self.all_threads:
            thread.start()

    def create_and_start_threads(self):
        
        self.create_threads()
        self.start_threads()

    @property
    def output_tensors(self):
        
        return self._output_tensors
