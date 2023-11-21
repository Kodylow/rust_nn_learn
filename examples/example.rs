use rust_nn_learn::*;
use std::env;
fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let neural_network_layers: Vec<usize> = vec![12288, 20, 7, 5, 1];
    let learning_rate = 0.0075;
    let iterations = 1000;
    let lambda = 0.0;

    let trainset = dataframe_from_csv("data/training_set.csv".into()).unwrap();
    let testset = dataframe_from_csv("data/test_set.csv".into()).unwrap();

    let training_data_array = array_from_dataframe(&trainset.data) / 255.0;
    let training_labels_array = array_from_dataframe(&trainset.labels);
    let test_data_array = array_from_dataframe(&testset.data) / 255.0;
    let test_labels_array = array_from_dataframe(&testset.labels);

    let model = NeuralNet {
        layers: neural_network_layers,
        learning_rate,
        lambda,
    };

    let parameters = model.initialize_parameters();

    let parameters = model.train_model(
        &training_data_array,
        &training_labels_array,
        parameters,
        iterations,
        model.learning_rate,
    );

    write_parameters_to_json_file(&parameters, "model.json".into());

    let training_predictions = model.predict(&training_data_array, &parameters);
    println!(
        "Training Set Accuracy: {}%",
        model.score(&training_predictions, &training_labels_array)
    );

    let test_predictions = model.predict(&test_data_array, &parameters);
    println!(
        "Test Set Accuracy: {}%",
        model.score(&test_predictions, &test_labels_array)
    );
}
