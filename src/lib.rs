use ndarray::prelude::*;
use polars::prelude::*;
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use std::{collections::HashMap, path::PathBuf};

#[derive(Debug)]
pub struct Dataset {
    data: DataFrame,
    labels: DataFrame,
}

/// Reads a CSV file and returns a tuple of two DataFrames for the training dataset and training labels.
pub fn dataframe_from_csv(file_path: PathBuf) -> PolarsResult<Dataset> {
    let data = CsvReader::from_path(file_path)?.has_header(true).finish()?;

    let (training_dataset, training_labels) = (data.drop("y")?, data.select(["y"])?);

    Ok(Dataset {
        data: training_dataset,
        labels: training_labels,
    })
}

/// Converts a DataFrame into a 2D array.
pub fn array_from_dataframe(df: &DataFrame) -> Array2<f32> {
    df.to_ndarray::<Float32Type>(IndexOrder::Fortran).unwrap()
}

/// A struct representing a neural network.
///
/// # Fields
///
/// * `layers` - A vector of usize where each element represents the number of neurons in a layer, normally `n`
/// * `learning_rate` - A float that represents the learning rate of the neural network, normally `h`
struct NeuralNet {
    pub layers: Vec<usize>,
    pub learning_rate: f32,
}

impl NeuralNet {
    /// Returns a HashMap dictionary of randomly initialized weights and biases for each layer in the neural network.
    ///
    /// # Returns
    ///
    /// * A HashMap where the keys are strings representing the layer and the type of parameter (weight or bias),
    ///   and the values are 2D arrays of the parameters themselves.
    pub fn initialize_parameters(&self) -> HashMap<String, Array2<f32>> {
        // Create an empty HashMap to store the parameters
        let mut parameters: HashMap<String, Array2<f32>> = HashMap::new();

        // Iterate over the layers of the neural network, starting from the second layer (index 1)
        for l in 1..self.layers.len() {
            // Initialize the weights and biases for the current layer
            let (weight_matrix, bias_matrix) = self.initialize_layer(l);

            // Insert the weight matrix into the parameters HashMap with a key indicating its layer and type
            parameters.insert(format!("W{}", l), weight_matrix);
            // Insert the bias matrix into the parameters HashMap with a key indicating its layer and type
            parameters.insert(format!("b{}", l), bias_matrix);
        }
        // Return the parameters HashMap
        parameters
    }

    /// Initialize a single layer with random weights and zero biases.
    ///
    /// # Arguments
    ///
    /// * `l` - The index of the layer to initialize.
    ///
    /// # Returns
    ///
    /// * A tuple containing the weight matrix and bias matrix for the layer.
    fn initialize_layer(&self, l: usize) -> (Array2<f32>, Array2<f32>) {
        // Create a uniform distribution between -1.0 and 1.0 for initializing weights
        let between = Uniform::from(-1.0..1.0);

        // Generate a vector of random weights. The size of the vector is determined by the number of neurons
        // in the current layer (`self.layers[l]`) and the number of neurons in the previous layer (`self.layers[l - 1]`).
        let mut rng = rand::thread_rng();
        let weight_array: Vec<f32> = (0..self.layers[l] * self.layers[l - 1])
            .map(|_| between.sample(&mut rng))
            .collect();

        // Generate a vector of biases, initialized to 0. The size of the vector is determined by the number of neurons
        // in the current layer (`self.layers[l]`).
        let bias_array: Vec<f32> = vec![0.0; self.layers[l]];

        // Convert the weight vector into a 2D array (matrix). The dimensions of the matrix are determined by the number of neurons
        // in the current layer (`self.layers[l]`) and the number of neurons in the previous layer (`self.layers[l - 1]`).
        let weight_matrix =
            Array::from_shape_vec((self.layers[l], self.layers[l - 1]), weight_array).unwrap();

        // Convert the bias vector into a 2D array (matrix). The dimensions of the matrix are determined by the number of neurons
        // in the current layer (`self.layers[l]`) and 1 (since each neuron has only one bias).
        let bias_matrix = Array::from_shape_vec((self.layers[l], 1), bias_array).unwrap();

        // Return the weight matrix and bias matrix as a tuple
        (weight_matrix, bias_matrix)
    }
}
