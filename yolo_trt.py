import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda  # noqa, must be imported
import pycuda.autoinit  # noqa, must be imported
# from sklearn import preprocessing
import common
trt.init_libnvinfer_plugins(None,'')

class YoloTRT(object):
    def _load_engine(self):
        with open(self.engine_file, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __init__(self, model,cuda_ctx=None):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.engine_file = model
        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        self.context = None
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        # lazy load implementation
        if self.engine is None:
            self.engine = self._load_engine()
            self.build()

    def build(self):
        # with open(self.engine_file, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
        #     self.engine = runtime.deserialize_cuda_engine(f.read())
        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = \
                            common.allocate_buffers(self.engine)
            
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()
    
    def __del__(self):
        """Free CUDA memories."""
        del self.outputs
        del self.inputs
        del self.stream

    def infer(self, objects_frame):
        batch_size = objects_frame.shape[0]
        allocate_place = np.prod(objects_frame.shape)
        self.inputs[0].host[:allocate_place] = objects_frame.flatten(order='C').astype(np.float32)
        
        if self.cuda_ctx:
            self.cuda_ctx.push()

        trt_outputs = common.do_inference(
            context=self.context, 
            bindings=self.bindings,
            inputs=self.inputs, 
            outputs=self.outputs, 
            stream=self.stream, 
            batch_size=batch_size)

        if self.cuda_ctx:
            self.cuda_ctx.pop()

        # embeddings = trt_outputs[0].reshape(-1, 512)
        # embeddings = embeddings[:batch_size]
        # # embeddings = preprocessing.normalize(embeddings)
        # return embeddings.copy()
        # print(trt_outputs[0].shape)
        return trt_outputs