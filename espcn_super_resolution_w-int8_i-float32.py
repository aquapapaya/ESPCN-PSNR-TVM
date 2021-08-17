"""
Deploy Pre-Trained TensorFlow Lite ESPCN
========================================
By Kuen-Wey Lin<kwlin@itri.org.tw>
"""

######################################################################
# Set environment variables
# -------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

import tvm
target = 'llvm'
target_host = 'llvm'
ctx = tvm.cpu(0)

model_path = './espcn_super_resolution_w-int8_i-float32.tflite'
input_name = 'input_1'
input_data_type = 'float32' # model input's data type
input_height = 300 # model input's height
input_width = 300 # model input's width
magnification = 3
img_path = './image/'
resulting_file_directory = './tvm_generated_files/'

######################################################################
# Load and pre-process test image
# -------------------------------

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import smart_resize
import numpy as np

def extract_YCbCr(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    reference_image = smart_resize(image, [input_height * magnification, input_width * magnification])

    shrunk_image = smart_resize(reference_image, [input_height, input_width])
    shrunk_image = array_to_img(shrunk_image)
    #shrunk_image.save('shrunk_image.jpg')
    image_ycbcr = shrunk_image.convert("YCbCr")
    luma, cb, cr = image_ycbcr.split()
    luma = img_to_array(luma)
    luma = luma.astype('float32') / 255.0
    luma = np.expand_dims(luma, axis=0)
    return reference_image, luma, cb, cr

######################################################################
# Load a TFLite model
# -------------------

tflite_model_file = os.path.join(model_path)
tflite_model_buf = open(tflite_model_file, "rb").read()

try:
    import tflite
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

######################################################################
# Convert the TFLite model into Relay IR
# --------------------------------------

import tvm.relay as relay
dtype_dict = {input_name: input_data_type}
shape_dict = {input_name: (1, input_height, input_width, 1)}
mod, params = relay.frontend.from_tflite(tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict)
print("Relay IR:\n", mod)

######################################################################
# Compile the Relay module
# ------------------------

with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize":True}):
    graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)

######################################################################
# Generate the five files for EV
# ------------------------------
'''
print("Printing host code to host_code.cc...")
with open('host_code.cc', 'w') as f:
    print(lib.get_source(), file=f)

print("Printing device code to device_code.cl...")
with open('device_code.cl', 'w') as f:
    print(lib.imported_modules[0].get_source(), file=f)

print("Printing meta json to device_code.tvm_meta.json...")
lib.imported_modules[0].save("device_code", "cl")
os.remove("device_code")

print("Printing binary parameters to binary_params.bin...")
with open('binary_params.bin', 'wb') as writer:
    writer.write(relay.save_param_dict(params))
    writer.close()

print("Printing graph to graph.json...")
with open('graph.json', 'w') as f:
    print(graph, file=f)
'''
######################################################################
# Move all resulting files to a directory
# ---------------------------------------
'''
import shutil

try:
    shutil.rmtree(resulting_file_directory)
except OSError as e:
    print("Preparing a directory for resulting files")

os.mkdir(resulting_file_directory)

shutil.move('kernel.txt', resulting_file_directory)
shutil.move('host_code.cc', resulting_file_directory)
shutil.move('device_code.cl', resulting_file_directory)
shutil.move('device_code.tvm_meta.json', resulting_file_directory)
shutil.move('binary_params.bin', resulting_file_directory)
shutil.move('graph.json', resulting_file_directory)
'''

######################################################################
# Create TVM runtime and do inference
# -----------------------------------

from tvm.contrib import graph_runtime
import PIL
import tensorflow as tf
def calculate_PSNR(graph, lib, params, ctx, file_name): # Peak signal-to-noise ratio
    # create module
    module = graph_runtime.create(graph, lib, ctx)

    # set input and parameters
    print('\nimage name:', file_name)
    reference_image, luma, cb, cr = extract_YCbCr(img_path + file_name)
    module.set_input(input_name, tvm.nd.array(luma))
    module.set_input(**params)

    # run
    import time
    timeStart = time.time()
    module.run()
    timeEnd = time.time()
    print("Inference time: %f" % (timeEnd - timeStart))

    # get output
    tvm_output = module.get_output(0).asnumpy()
    output_luma = tvm_output[0,:,:,0]
    output_luma *= 255.0
    output_luma = PIL.Image.fromarray(np.uint8(output_luma.clip(0, 255)), mode='L')
    output_cb = cb.resize(output_luma.size, PIL.Image.BICUBIC)
    output_cr = cr.resize(output_luma.size, PIL.Image.BICUBIC)
    enlarged_image = PIL.Image.merge('YCbCr', [output_luma, output_cb, output_cr]).convert('RGB')
    #enlarged_image.save('enlarged_image.jpg')
    psnr = tf.image.psnr(img_to_array(enlarged_image, dtype='uint8'), img_to_array(reference_image, dtype='uint8'), max_val=255)
    print('PSNR:', psnr.numpy(), 'dB')
    return psnr.numpy()

print("\nStart to run model using TVM and then calculate PSNR...")
total_psnr = 0
file_list = os.listdir(img_path)
for file_name in file_list:
    total_psnr += calculate_PSNR(graph, lib, params, ctx, file_name)
print("\nNum of tested images:", len(file_list))
average_psnr =  total_psnr/len(file_list)
print("Average PSNR: ", round(average_psnr, 2), 'dB')
