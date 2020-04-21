	• The QKeras model as produced by qkeras is: 7_4_2020_quantized.onnx 

	• The model as produced by normal Keras is: integer_scouting_7_4_2020_3layers.onnx

	• The script to convert from standard Keras to onnx is: Model_to_onnx.py
	
	• The script to convert from qkeras to onnx is: qkeras_to_onnx.py. To do this we use tf2onnx(ver 1.6). 

	• The script for inference is: scouting_model_comparison_PvsB_working.py

	• For training with qkeras we use this script: QKeras_testing.py

	• For training with regular keras:  integer_scouting_model.py
	
	• ie_20Events_output_withoutV.txt: Inference on the board and in onnxruntime session to compare the outputs of 200 
		instances (20 per 20 events), without "options","V"
	
	• ie_20Events_output_withV.txt: Inference on the board and in onnxruntime session to compare the outputs of 200 
		instances (20 per 20 events), with "options","V"
	
	
