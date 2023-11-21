use ndarray::prelude::*;
use polars::prelude::*;
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use std::f32::consts::E;
use std::fs::OpenOptions;
use std::{collections::HashMap, path::PathBuf};

#[derive(Debug)]
pub struct Dataset {
    pub data: DataFrame,
    pub labels: DataFrame,
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
/// need to reverse the axes because polars uses row-major order and ndarray uses column-major order
pub fn array_from_dataframe(df: &DataFrame) -> Array2<f32> {
    df.to_ndarray::<Float32Type>(IndexOrder::Fortran)
        .unwrap()
        .reversed_axes()
}

/// A struct representing a neural network.
///
/// # Fields
///
/// * `layers` - A vector of usize where each element represents the number of neurons in a layer, normally `n`
/// * `learning_rate` - A float that represents the learning rate of the neural network, normally `h`
pub struct NeuralNet {
    pub layers: Vec<usize>,
    pub learning_rate: f32,
    pub lambda: f32,
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

    pub fn update_parameters(
        &self,
        params: &HashMap<String, Array2<f32>>,
        grads: HashMap<String, Array2<f32>>,
        m: f32,
        learning_rate: f32,
    ) -> HashMap<String, Array2<f32>> {
        let mut parameters = params.clone();
        let num_of_layers = self.layers.len() - 1;
        for l in 1..num_of_layers + 1 {
            let weight_string_grad = ["dW", &l.to_string()].join("").to_string();
            let bias_string_grad = ["db", &l.to_string()].join("").to_string();
            let weight_string = ["W", &l.to_string()].join("").to_string();
            let bias_string = ["b", &l.to_string()].join("").to_string();

            *parameters.get_mut(&weight_string).unwrap() = parameters[&weight_string].clone()
                - (learning_rate
                    * (grads[&weight_string_grad].clone()
                        + (self.lambda / m) * parameters[&weight_string].clone()));
            *parameters.get_mut(&bias_string).unwrap() = parameters[&bias_string].clone()
                - (learning_rate * grads[&bias_string_grad].clone());
        }
        parameters
    }

    /// Performs the forward propagation process for the entire neural network.
    pub fn forward(
        &self,
        x: &Array2<f32>,
        parameters: &HashMap<String, Array2<f32>>,
    ) -> (Array2<f32>, HashMap<String, (LinearCache, ActivationCache)>) {
        let number_of_layers = self.layers.len() - 1;

        // Initializes the a matrix as a copy of x and creates an empty hashmap caches to store the caches for each layer.
        let mut a = x.clone();
        let mut caches = HashMap::new();

        // Performs the linear forward activation function for each layer, using the ReLU activation function for all layers except the last.
        for l in 1..number_of_layers {
            let w_string = ["W", &l.to_string()].join("").to_string();
            let b_string = ["b", &l.to_string()].join("").to_string();

            let w = &parameters[&w_string];
            let b = &parameters[&b_string];

            let (a_temp, cache_temp) =
                linear_forward_activation(&a, w, b, Activation::Relu).unwrap();

            a = a_temp;

            caches.insert(l.to_string(), cache_temp);
        }

        // Compute activation of last layer with sigmoid
        let weight_string = ["W", &(number_of_layers).to_string()].join("").to_string();
        let bias_string = ["b", &(number_of_layers).to_string()].join("").to_string();

        let w = &parameters[&weight_string];
        let b = &parameters[&bias_string];

        let (al, cache) = linear_forward_activation(&a, w, b, Activation::Sigmoid).unwrap();
        caches.insert(number_of_layers.to_string(), cache);

        return (al, caches);
    }

    /// Measures the discrepancy between the predicted labels and the true labels using the cross-entropy loss function.
    ///
    /// al = predicted labels
    /// y = true labels
    /// m = number of training examples
    pub fn cost(&self, al: &Array2<f32>, y: &Array2<f32>) -> f32 {
        let m = y.shape()[1] as f32;
        let cost = -(1.0 / m)
            * (y.dot(&al.clone().reversed_axes().log())
                + (1.0 - y).dot(&(1.0 - al).reversed_axes().log()));

        return cost.sum();
    }

    pub fn backward(
        &self,
        al: &Array2<f32>,
        y: &Array2<f32>,
        caches: HashMap<String, (LinearCache, ActivationCache)>,
    ) -> HashMap<String, Array2<f32>> {
        let mut grads = HashMap::new();
        let num_of_layers = self.layers.len() - 1;

        let dal = -(y / al - (1.0 - y) / (1.0 - al));

        let current_cache = caches[&num_of_layers.to_string()].clone();
        let (mut da_prev, mut dw, mut db) =
            linear_backward_activation(&dal, current_cache, Activation::Sigmoid);

        let weight_string = ["dW", &num_of_layers.to_string()].join("").to_string();
        let bias_string = ["db", &num_of_layers.to_string()].join("").to_string();
        let activation_string = ["dA", &num_of_layers.to_string()].join("").to_string();

        grads.insert(weight_string, dw);
        grads.insert(bias_string, db);
        grads.insert(activation_string, da_prev.clone());

        for l in (1..num_of_layers).rev() {
            let current_cache = caches[&l.to_string()].clone();
            (da_prev, dw, db) =
                linear_backward_activation(&da_prev, current_cache, Activation::Relu);

            let weight_string = ["dW", &l.to_string()].join("").to_string();
            let bias_string = ["db", &l.to_string()].join("").to_string();
            let activation_string = ["dA", &l.to_string()].join("").to_string();

            grads.insert(weight_string, dw);
            grads.insert(bias_string, db);
            grads.insert(activation_string, da_prev.clone());
        }

        grads
    }

    /// Trains the neural network using the training data and labels.
    /// Returns a HashMap of the optimized parameters and prints the cost every 100 iterations (epoch)
    pub fn train_model(
        &self,
        x_train_data: &Array2<f32>,
        y_train_data: &Array2<f32>,
        mut parameters: HashMap<String, Array2<f32>>,
        iterations: usize,
        learning_rate: f32,
    ) -> HashMap<String, Array2<f32>> {
        let mut costs: Vec<f32> = vec![];

        for i in 0..iterations {
            let (al, caches) = self.forward(&x_train_data, &parameters);
            let cost = self.cost(&al, &y_train_data);
            let grads = self.backward(&al, &y_train_data, caches);
            parameters = self.update_parameters(
                &parameters,
                grads.clone(),
                y_train_data.shape()[1] as f32,
                learning_rate,
            );

            if i % 100 == 0 {
                costs.append(&mut vec![cost]);
                println!("Epoch : {}/{}    Cost: {:?}", i, iterations, cost);
            }
        }
        parameters
    }

    /// 1. Calls the forward method with the test data and the optimized parameters
    /// to obtain the final activation al.
    /// 2. Applies a threshold of 0.5 to the elements of al using the map method,
    /// converting values greater than 0.5 to 1.0 and values less than or equal to 0.5 to 0.0.
    /// 3. Returns the predicted labels as y_hat.
    pub fn predict(
        &self,
        x_test_data: &Array2<f32>,
        parameters: &HashMap<String, Array2<f32>>,
    ) -> Array2<f32> {
        let (al, _) = self.forward(&x_test_data, &parameters);

        let y_hat = al.map(|x| (x > &0.5) as i32 as f32);
        y_hat
    }

    /// Calculates the accuracy score of the predicted labels compared to the actual test labels.
    pub fn score(&self, y_hat: &Array2<f32>, y_test_data: &Array2<f32>) -> f32 {
        let error =
            (y_hat - y_test_data).map(|x| x.abs()).sum() / y_test_data.shape()[1] as f32 * 100.0;
        100.0 - error
    }
}

/// LinearCache stores the intermediate values needed for each layer
#[derive(Clone, Debug)]
pub struct LinearCache {
    /// The activation matrix
    pub a: Array2<f32>,
    /// The weight matrix
    pub w: Array2<f32>,
    /// The bias matrix
    pub b: Array2<f32>,
}

/// The sigmoid function takes a single value z as input and returns the sigmoid activation,
/// which is calculated using the sigmoid formula: 1 / (1 + e^-z).
///
/// The sigmoid function maps the input value to a range between 0 and 1,
/// enabling the network to model non-linear relationships.
pub fn sigmoid(z: &f32) -> f32 {
    1.0 / (1.0 + E.powf(-z))
}

/// The relu function takes a single value z as input and applies the Rectified Linear Unit (ReLU) activation.
/// If z is greater than zero, the function returns z; otherwise, it returns zero.
/// ReLU is a popular activation function that introduces non-linearity and helps the network learn complex patterns.
pub fn relu(z: &f32) -> f32 {
    match *z > 0.0 {
        true => *z,
        false => 0.0,
    }
}

/// ActivationCache stores the logit matrix z for each layer
#[derive(Clone, Debug)]
pub struct ActivationCache {
    pub z: Array2<f32>,
}

#[derive(Clone, Debug, Copy)]
pub enum Activation {
    Sigmoid,
    Relu,
}

/// Applies the specified activation function to a 2D matrix z.
/// The resulting activation matrix is returned along with an ActivationCache struct that stores the corresponding logit matrix.
pub fn activation(z: Array2<f32>, activation: Activation) -> (Array2<f32>, ActivationCache) {
    match activation {
        Activation::Sigmoid => (z.mapv(|x| sigmoid(&x)), ActivationCache { z }),
        Activation::Relu => (z.mapv(|x| relu(&x)), ActivationCache { z }),
    }
}

/// Takes the activation matrix a, weight matrix w, and bias matrix b as inputs.
/// It performs the linear transformation by calculating the dot product of w and a, and then adding b to the result.
/// The resulting matrix z represents the logits of the layer.
/// The function returns z along with a LinearCache struct that stores the input matrices for later use in backprop.
pub fn linear_forward(
    a: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array2<f32>,
) -> (Array2<f32>, LinearCache) {
    let z = w.dot(a) + b;

    let cache = LinearCache {
        a: a.clone(),
        w: w.clone(),
        b: b.clone(),
    };
    return (z, cache);
}

/// Builds upon the linear_forward function.
/// It takes the same input matrices as linear_forward,
/// along with an additional activation parameter indicating the activation function to be applied.
/// The function first calls linear_forward to obtain the logits z and the linear cache.
/// Then, depending on the specified activation function,
/// it calls either sigmoid_activation or relu_activation to compute the activation matrix a_next and the activation cache.
/// The function returns a_next along with a tuple of the linear cache and activation cache, wrapped in a Result enum.
pub fn linear_forward_activation(
    a: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array2<f32>,
    actv: Activation,
) -> Result<(Array2<f32>, (LinearCache, ActivationCache)), String> {
    let (z, linear_cache) = linear_forward(a, w, b);
    let (a_next, activation_cache) = activation(z, actv);
    return Ok((a_next, (linear_cache, activation_cache)));
}

/// Calculates the derivative of the sigmoid activation function.
/// It takes the input z and returns the derivative value,
/// which is computed as the sigmoid of z multiplied by 1.0 minus the sigmoid of z.
pub fn sigmoid_prime(z: &f32) -> f32 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

/// Computes the derivative of the ReLU activation function.
/// It takes the input z and returns 1.0 if z is greater than 0, and 0.0 otherwise.
pub fn relu_prime(z: &f32) -> f32 {
    match *z > 0.0 {
        true => 1.0,
        false => 0.0,
    }
}

/// Calculates the derivative of the specified activation function.
pub fn activation_prime(z: &f32, activation: Activation) -> f32 {
    match activation {
        Activation::Sigmoid => sigmoid(z) * (1.0 - sigmoid(z)),
        Activation::Relu => match *z > 0.0 {
            true => 1.0,
            false => 0.0,
        },
    }
}

/// Calculates the backward propagation for the specified activation function.
/// It takes the derivative of the cost function with respect to the activation da and the activation cache activation_cache.
/// It performs an element-wise multiplication between da and the derivative of the activation function
/// applied to the values in the activation cache, activation_cache.z.
pub fn activation_backward(
    da: &Array2<f32>,
    activation_cache: ActivationCache,
    actv: Activation,
) -> Array2<f32> {
    da * activation_cache.z.mapv(|x| activation_prime(&x, actv))
}

/// Calculates the backward propagation for the linear component of a layer.
pub fn linear_backward(
    dz: &Array2<f32>,
    linear_cache: LinearCache,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    // Extract the previous layer’s activation a_prev, the weight matrix w, and the bias matrix _b from the linear cache
    let (a_prev, w, _b) = (linear_cache.a, linear_cache.w, linear_cache.b);

    // Compute the number of training examples m
    let m = a_prev.shape()[1] as f32;

    // Calculate the gradient of the weights dw using the dot product between dz and the transposed a_prev, scaled by 1/m
    let dw = (1.0 / m) * (dz.dot(&a_prev.reversed_axes()));

    // Compute the gradient of the biases db by summing the elements of dz along Axis(1) and scaling the result by 1/m
    let db_vec = ((1.0 / m) * dz.sum_axis(Axis(1))).to_vec();
    let db = Array2::from_shape_vec((db_vec.len(), 1), db_vec).unwrap();

    // Compute the gradient of the previous layer's activation da_prev by performing the dot product between the transposed w and dz
    let da_prev = w.reversed_axes().dot(dz);

    // Return the gradients with respect to the previous layer's activation da_prev, the weights dw, and the biases db
    (da_prev, dw, db)
}

/// Builds upon the linear_backward function.
/// takes something like: &da_prev, current_cache, "relu");
pub fn linear_backward_activation(
    da: &Array2<f32>,
    cache: (LinearCache, ActivationCache),
    activation: Activation,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (linear_cache, activation_cache) = cache;

    let dz = activation_backward(da, activation_cache, activation);

    let (da_prev, dw, db) = linear_backward(&dz, linear_cache);

    (da_prev, dw, db)
}

trait Log {
    fn log(&self) -> Array2<f32>;
}

impl Log for Array2<f32> {
    fn log(&self) -> Array2<f32> {
        self.mapv(|x| x.log(std::f32::consts::E))
    }
}

pub fn write_parameters_to_json_file(
    parameters: &HashMap<String, Array2<f32>>,
    file_path: PathBuf,
) {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(file_path)
        .unwrap();

    _ = serde_json::to_writer(file, parameters);
}
