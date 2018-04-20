# ConveyLife
Conway's Game of Life implementation using TenserFlow.

[TensorFlow](https://www.tensorflow.org/install/) should be installed on your machine.

Use [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) (included in TensorFlow) to visualize computation graph.

You can run the script with one of predefined or your own map of cells as a text file:<br/>
`python ConwayLife.py <cellsmapfile.txt>`

To run the script with random map of desired size and number of cells alive, specify three numeric parameters:<br/>
`python ConwayLife.py <width> <height> <aliveCellsNumber>`
  
To run the script on completely random map, no parameters should be specified:<br/>
`python ConwayLife.py`
