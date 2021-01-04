import tensorflow as tf

#@title Select Quantization Strategy
model_checkpoint_name = "ssd_mobiledet_cpu_coco" 
model_to_be_quantized = "ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/tflite_graph.pb"
quantization_strategy = "fp16" #@param ["dr", "fp16", "int8"]

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file=model_to_be_quantized, 
    input_arrays=['normalized_input_image_tensor'],
    output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'],
    input_shapes={'normalized_input_image_tensor': [1, 320, 320, 3]}
)
converter.allow_custom_ops = True
if quantization_strategy=="dr":
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
elif quantization_strategy=="fp16":
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
elif quantization_strategy=="int8":
    converter.inference_input_type = tf.uint8
    converter.quantized_input_stats = {"normalized_input_image_tensor": (128, 128)}
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

tflite_filename = model_checkpoint_name + "_" + quantization_strategy + ".tflite"
open(tflite_filename, 'wb').write(tflite_model)
print(f"TFLite model generated with {quantization_strategy}")