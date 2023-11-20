use std::path::PathBuf;

use anyhow::Result;
use rust_nn_learn::dataframe_from_csv;
fn main() -> Result<()> {
    let train_dataset = dataframe_from_csv(PathBuf::from("data/training_set.csv"))?;
    println!("train_data: {:#?}", train_dataset);

    Ok(())
}
