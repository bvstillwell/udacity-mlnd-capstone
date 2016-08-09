echo "Freezing ./save/$@.pb"

~/github/tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph \
	--input_graph=save/$@.pb \
	--input_checkpoint=save/$@.ckpt-0 \
	--output_graph=save/frozen_graph.pb \
	--output_node_names=tf_predict_single_output

mkdir -p android/assets
cp save/frozen_graph.pb android/assets/
ls -lh save/frozen_graph.pb
echo "Done!"
