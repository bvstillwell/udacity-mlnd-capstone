~/github/tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph \
	--input_graph=save/model.pb \
	--input_checkpoint=save/model.ckpt-0 \
	--output_graph=save/frozen_graph.pb \
	--output_node_names=predict_single_output

cp save/model.pb ../android/assets/