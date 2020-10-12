requirements: 
tensorflow-gpu==2.0.0
scikit-image==0.16.2
opencv==3.3.1

Class STROKE_MASK, takes path to weights foler to initialize.
weights_folder: The path to the folder containing: "checkpoint", "model.tf.index".... Provided in this folder.

Predict function:
"""
        :param tif_stack_list: A list of numpy arrays, each array is a stack.
        :param heatmap: if true, Return the original predictions, has the same shape as input.
                        axis 0 => background, 1 => Normal, 2 => Stroke
                        The number in each axis indicates the confidence value.
                        [
                            (steps, height, width, 3),
                            (steps, height, width, 3),
                            ...
                        ]
                        If False, Return the one-hot label of predictions.
                        [
                            (steps, height, width, 3), [1, 0, 0] => background, [0, 1, 0] => Normal, [0, 0, 1] => Stroke
                            (steps, height, width, 3),
                            ...
                        ]
        :param color: Covert one-hot predictions to color mask.
        :return: A list of predictions
        """

