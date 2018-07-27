+++
title = "Recommending books (with Rust)"
author = ["maciej"]
date = 2018-07-27T09:17:00-07:00
lastmod = 2018-07-27T09:18:32-07:00
categories = ["engineering"]
draft = false
weight = 2001
math = true
+++

In this post, we're going to build a sequence-based recommender system in Rust: a sytem that accepts a person's reading history as input, and outputs recommendations on what to read next.

Building systems like this -- like much of machine learning and data science -- is normally the province of Python. The combination of numpy, pandas, and other libraries that build on them makes doing data science in Python a breeze.

Nevertheless, there are advantages of using statically typed programming languages for some machine learning tasks: help from the compiler, better self-documentation capacity, and speed of the resulting code. This is part of the motivation behind building [Tensorflow for Swift](https://github.com/tensorflow/swift): modern statically typed programming languages with good ergonomics, an expressive type system, and value types bring a lot of advantages.

In this blog post, I hope to show that these advantages do not come at the cost of verbose code and complexity that distracts from the main task. In many ways, Rust comes very close to the ergonomics and expressiveness of Python.

You can read this post like a Jupyter notebook: a series of steps that builds into a complete program. To see the end result, jump to the appendix, or have a look at the [Github repository](https://github.com/maciejkula/hugo-blog/tree/master/code/goodbooks-recommender/).


## Setting up a project {#setting-up-a-project}

Rust projects follow a certain structure, and we can set up a new project using `cargo`:

```bash
cargo new --bin goodbooks-recommender
```

This will set up a new directory (and a git repository), containing a `Cargo.toml` file with information about the package, and a `main.rs` file that contains our code. To run the code, run

```bash
cargo run
```

This will download the dependencies, compile the code, and run the binary all in one. (We will want to run `cargo run --release` once we start fitting models: this turns on the optimization passes in the compile step.)


## Getting the data {#getting-the-data}

To train the model, we'll use the [Goodbooks-10k dataset](https://github.com/zygmuntz/goodbooks-10k): a dataset of approximately 6 million ratings over 10,000 books from over 50,000 users, derived from the popular Goodreads service.

The first dependency we are going to use is `reqwest`: a crate similar to Python's `requests` that will allow us to easily download the data we need. The second is `failure`, a crate that makes dealing with errors easier.


### Downloading data {#downloading-data}

With its help, we can start defining our download function:

<a id="org22ba2dc"></a>
```rust

// Need to import a couple of things from
// the standard library
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Download file from `url` and save it to `destination`.
fn download(url: &str, destination: &Path)
            -> Result<(), failure::Error> {

    // Don't do anything if we already have the file.
    if destination.exists() {
        return Ok(())
    }

    // Otherwise, create a new file.

    // Because each of the following operations
    // can fail (returns a result type), we follow
    // them with the `?` operator. If the result
    // is an error, it will exit from the function
    // early, propagating the error upwards; if
    // the operation completed successfully, we get
    // the result instead.
    let file = File::create(destination)?;
    let mut writer = BufWriter::new(file);

    let mut response = reqwest::get(url)?;
    response.copy_to(&mut writer)?;

    Ok(())
}
```

With this, we can write a short function that downloads both the ratings file and a file that contains metadata on the books from the dataset:

<a id="orgdcc3c2a"></a>
```rust
/// Download ratings and metadata both.
fn download_data(ratings_path: &Path, books_path: &Path) {
    let ratings_url = "https://github.com/zygmuntz/\
                       goodbooks-10k/raw/master/ratings.csv";
    let books_url = "https://github.com/zygmuntz/\
                     goodbooks-10k/raw/master/books.csv";

    download(&ratings_url,
             ratings_path).expect("Could not download ratings");
    download(&books_url,
             books_path).expect("Could not download metadata");
}
```

The ratings file looks like this:

```text
user_id,book_id
9,8
15,398
15,275
37,7173
34,380
```


### Parsing {#parsing}

We have two options for parsing the resulting CSV files. One is to parse things manually; the other, to use Rust's amazing serialization/deserialization capabilities and the [`csv` crate](https://crates.io/crates/csv).

The heart of Rust's serialization ecosystem lies in the [`serde` crate](https://serde.rs/). It provides traits that allow structs to be seamlessly serialized and deserialized across a variety for formats. We'll derive those on a `WishlistEntry` struct to be able to read it from the CSV file:

<a id="org7962a11"></a>
```rust
// Importing this allows us to autoderive
// the serialization traits.
#[macro_use]
extern crate serde_derive;

// This is where we get the serde traits from.
extern crate serde;

// An implementation of the serde encoders/decoders
// to and from a JSON. We'll need
// these later.
extern crate serde_json;

#[derive(Debug, Serialize, Deserialize)]
struct WishlistEntry {
    user_id: usize,
    book_id: usize,
}
```

After importing the `csv` crate we're ready to write the deserialize function:

<a id="orga3c7e72"></a>
```rust
extern crate csv;

/// Deserialize from file at `path` into a vector of
/// `WishlistEntry`.
fn deserialize_ratings(path: &Path)
               -> Result<Vec<WishlistEntry>, failure::Error> {

    let mut reader = csv::Reader::from_path(path)?;

    // We specify the type of the deserialized entity
    // via a type annotation. Otherwise, the compiler has
    // no way of knowing what sort of thing we want to
    // deserialize!
    // We also do a further trick where instead of deserializing
    // into a vector of results, we deserialize into a result with
    // a vector.
    let entries: Vec<WishlistEntry> = reader.deserialize()
        .collect::<Result<Vec<_>, _>>()?;

    Ok(entries)
}
```

We also want to deserialize the metadata. We're only really interested in the book id and title, as this is what will allow us to make and evaluate recommendations based on titles rather than book ids.

As before, we define a struct and a corresponding deserialize function. This time, we are going to return two mappings instead of a vector: the first mapping book ids to book titles, the second book titles to book ids.

<a id="org2666e67"></a>
```rust
#[derive(Debug, Deserialize, Serialize)]
struct Book {
    book_id: usize,
    title: String
}

// We'll use the stdlib hashmap for the mapping.
use std::collections::HashMap;

/// Deserialize from file at `path` into the book
/// mappings.
fn deserialize_books(path: &Path)
   -> Result<(HashMap<usize, String>,
              HashMap<String, usize>), failure::Error> {

    let mut reader = csv::Reader::from_path(path)?;

    let entries: Vec<Book> = reader.deserialize::<Book>()
        .collect::<Result<Vec<_>, _>>()?;

    // We can simply iterate over the entries and collect
    // them into a different data structure. This is not
    // the most efficient solution but it will do for now.
    let id_to_title: HashMap<usize, String> = entries
        .iter()
        .map(|book| (book.book_id, book.title.clone()))
        .collect();
    let title_to_id: HashMap<String, usize> = entries
        .iter()
        .map(|book| (book.title.clone(), book.book_id))
        .collect();

    Ok((id_to_title, title_to_id))
}
```


## Fitting a model {#fitting-a-model}

Now that we have read the data, we can start thinking about what models to fit, and how to fit them.

The [`sbr`](https://github.com/maciejkula/sbr-rs) package implements two recommender models:

-   an LSTM-based model, and
-   an exponential moving average (EWMA) model.

The first is much more powerful: it implements a full LSTM model, taking a user's history of past interactions and trying to predict their next action.

The second is simpler computationally: the user representation at time \\(t\\), \\(u\_t\\) , is simply an exponentially weighted average of \\(i\_t\\), the ($d$-dimensional) embeddings of items the user interacted with at time \\(t\\):
\\[
   u\_t = (1 - \sigma(\alpha))u\_{t-1} + \sigma(\alpha)i\_t,
\\]
where \\(\sigma(\alpha)\\) is the exponential averaging weight, rescaled to lie between 0 and via the sigmoid function \\(\sigma\\).

Despite its simplicity, the model seems to perform fairly well on the Movielens dataset, and we're going to use it for this example.


### Setting up hyperparameters {#setting-up-hyperparameters}

The first thing we need to do is to write a function that will set up all the hyperparameters of the model:

<a id="org94e46d9"></a>
```rust
extern crate sbr;

use sbr::models::ewma::{Hyperparameters, ImplicitEWMAModel};
use sbr::models::{Loss, Optimizer};

fn build_model(num_items: usize) -> ImplicitEWMAModel {
    let hyperparameters = Hyperparameters::new(num_items, 128)
        .embedding_dim(32)
        .learning_rate(0.16)
        .l2_penalty(0.0004)
        .loss(Loss::WARP)
        .optimizer(Optimizer::Adagrad)
        .num_epochs(10)
        .num_threads(1);

    hyperparameters.build()
}
```


### Preparing data {#preparing-data}

The second is to convert the `WishlistEntry` objects into `sbr`'s [`Interaction`](https://docs.rs/sbr/0.4.0/sbr/data/struct.Interactions.html) objects:

<a id="org9524419"></a>
```rust
use sbr::data::{Interaction, Interactions};

fn build_interactions(data: &[WishlistEntry]) -> Interactions {
    // If the collection is empty, `max` doesn't exist. This
    // is why we get an Option back, which we then unwrap.
    let num_users = data
        .iter()
        .map(|x| x.user_id)
        .max()
        .unwrap() + 1;
    let num_items = data
        .iter()
        .map(|x| x.book_id)
        .max()
        .unwrap() + 1;

    let mut interactions = Interactions::new(num_users,
                                             num_items);

    // There are no timestamps in the interaction data, but
    // we make use of the fact that they are sorted by time.
    for (idx, datum) in data.iter().enumerate() {
        interactions.push(
            Interaction::new(datum.user_id,
                             datum.book_id,
                             idx)
        );
    }

    interactions
}
```


### Fitting {#fitting}

The model fitting itself is easy: we've set up the data and hyperparameters, and all that is left is to fit the model, making sure we have a train-test split to evaluate performance:

<a id="org930b43a"></a>
```rust
// We need to import the rand crate.
extern crate rand;
use rand::SeedableRng;

// We perform a split where the train and test
// sets are disjoint on the user dimension: no
// single user is in both.
use sbr::data::user_based_split;
use sbr::OnlineRankingModel;

use sbr::evaluation::mrr_score;

/// Fit the model.
///
/// If successful, return the MRR on the test set.
/// Otherwise, return an error.
fn fit(model: &mut ImplicitEWMAModel,
       data: &Interactions)
       -> Result<f32, failure::Error> {

    // Use a fixed seed for repeatable results.
    let mut rng = rand::XorShiftRng::from_seed([42; 16]);

    let (train, test) = user_based_split(data,
                                         &mut rng,
                                         0.2);

    model.fit(&train.to_compressed())?;

    let mrr = mrr_score(model, &test.to_compressed())?;

    Ok(mrr)
}

```

On my machine, this takes about a minute and a half, and achieves an MRR of 0.09. This is an OK result. To improve it, we could perform a hyperparameter search --- the `Hyperparameters` struct has a [`random`](https://docs.rs/sbr/0.4.0/sbr/models/ewma/struct.Hyperparameters.html#method.random) constructor that facilitates this. For now, however, we'll stick with this what we have.

Once we have the model, we'll want to save it for future use. Again, we'll use the `serde` library to do so:

<a id="org77278a5"></a>
```rust
fn serialize_model(model: &ImplicitEWMAModel,
                   path: &Path) -> Result<(), failure::Error> {

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    Ok(serde_json::to_writer(&mut writer, model)?)
}
```

Wiring all the bits together gives

<a id="org6dfbf0f"></a>
```rust
/// Download training data and build a model.
///
/// We'll use this function to power the `fit`
/// subcommand of our command line tool.
fn main_build() {

    let ratings_path = Path::new("ratings.csv");
    let books_path = Path::new("books.csv");
    let model_path = Path::new("model.json");

    // Exit early if we already have a model.
    if model_path.exists() {
        println!("Model already fitted.");
        return ();
    }

    download_data(ratings_path, books_path);

    let ratings = deserialize_ratings(ratings_path).unwrap();
    let (id_to_title,
         title_to_id) = deserialize_books(books_path).unwrap();

    println!("Deserialized {} ratings.", ratings.len());
    println!("Deserialized {} books.", id_to_title.len());

    let interactions = build_interactions(&ratings);
    let mut model = build_model(interactions.num_items());

    println!("Fitting...");
    let mrr = fit(&mut model, &interactions)
        .expect("Unable to fit model.");
    println!("Fit model with MRR of {:.2}", mrr);

    serialize_model(&model, &model_path)
        .expect("Unable to serialize model.");
}
```


## Getting predictions {#getting-predictions}

We need two bits here: (1) deserializing the model, and (2) getting predictions.

For the first, the following should suffice:

<a id="org7bfd96e"></a>
```rust
use std::io::BufReader;

fn deserialize_model() -> Result<ImplicitEWMAModel,
                                 failure::Error> {

    let file = File::open("model.json")?;
    let reader = BufReader::new(file);

    let model = serde_json::from_reader(reader)?;

    Ok(model)
}
```

For the second, we'll accept a sequence of book titles as input, translate to indices, get predictions, and translate back to book titles.

<a id="org3dbfef8"></a>
```rust
fn predict(input_titles: &[String],
           model: &ImplicitEWMAModel)
           -> Result<Vec<String>, failure::Error> {
    let (id_to_title,
         title_to_id) = deserialize_books(
        &Path::new("books.csv")
    ).unwrap();

    // Let's first check if the inputs are valid.
    for title in input_titles {
        if !title_to_id.contains_key(title) {
            println!("No such title, ignoring: {}", title);
        }
    }

    // Map the titles to indices.
    let input_indices: Vec<_> = input_titles
        .iter()
        .filter_map(|title| title_to_id.get(title))
        .cloned()
        .collect();
    let indices_to_score: Vec<usize> =
        (0..id_to_title.len()).collect();

    // Get the user representation.
    let user = model.user_representation(&input_indices)?;
    // Get the actual predictions.
    let predictions = model.predict(&user, &indices_to_score)?;

    // We implement argsort by zipping item indices
    // with their scores into tuples...
    let mut predictions: Vec<_>
        = indices_to_score.iter()
        .zip(predictions)
        .map(|(idx, score)| (idx, score))
        .collect();

    // ...and sorting the result in descending order.
    // This is a little tricky for floats are they
    // are not always comparable (they could be NaN or Inf),
    // so we use partial sorting and fail the program
    // if non-finite values are encountered.
    predictions
        .sort_by(|(_, score_a), (_, score_b)|
                 score_b.partial_cmp(score_a)
                 .unwrap());

    // Finally, we get the names for the top 10 items.
    Ok((&predictions[..10])
       .iter()
       .map(|(idx, _)| id_to_title.get(idx).unwrap())
       .cloned()
       .collect())
}
```


## Putting it all together {#putting-it-all-together}

Finally, we can write our `main` function. It'll look at the command line arguments and call either the model building or the prediction functions.

<a id="org970da27"></a>
```rust
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        println!("First argument must be \
                  one of 'fit' or 'predict'.");
        return ();
    }

    // We need to convert a `String` into a
    // `&str` here. This is one of the few
    // cases where Rust's ergonomics still
    // have some way to go.
    match &args[0][..] {
        "fit" => main_build(),
        "predict" => {
            let model = deserialize_model()
                .expect("Unable to deserialize model.");
            let predictions = predict(&args[1..], &model)
                .expect("Unable to get predictions");
            println!("Predictions:");
            for prediction in predictions {
                println!("    {}", prediction);
            }
        },
        _ => println!("First argument must be \
                       one of 'fit' or 'predict'."),
    }
}
```

What about the results? They look reasonable at first blush if you are a fan of the Harry Potter series:

```shell
time cargo run --release -- predict "Harry Potter and the Order of the Phoenix (Harry Potter, #5, Part 1)"
  Finished release [optimized] target(s) in 0.12s
Predictions:
    Harry Potter and the Order of the Phoenix (Harry Potter, #5, Part 1)
    Harry Potter and the Prisoner of Azkaban (Harry Potter, #3)
    Quidditch Through the Ages
    Harry Potter and the Goblet of Fire (Harry Potter, #4)
    Harry Potter and the Sorcerer's Stone (Harry Potter, #1)
    Harry Potter: Film Wizardry
    The Harry Potter Collection 1-4 (Harry Potter, #1-4)
    Harry Potter and the Chamber of Secrets (Harry Potter, #2)
    Harry Potter and the Deathly Hallows (Harry Potter, #7)
    Harry Potter and the Order of the Phoenix (Harry Potter, #5)

```

If you prefer Faulkner, the results are relatively sensible too:

```shell
time cargo run --release -- predict "As I Lay Dying"
Predictions:
    As I Lay Dying
    A Portrait of the Artist as a Young Man
    The Sound and the Fury
    Death of a Salesman
    The Things They Carried
    The Awakening
    Invisible Man
    A Separate Peace
    The House on Mango Street
    The Glass Menagerie
```

We've got a working model. Of course, serving recommendations via a CLI tool is not very useful: ideally, we'd have a web service that can serve these more widely. This, however, will have to wait for another blog post.


## Appendix {#appendix}

The final result looks like this:


### Cargo.toml {#cargo-dot-toml}

<a id="org3d195ca"></a>
```text
[package]
name = "goodbooks-recommender"
version = "0.1.0"
authors = ["Maciej Kula"]

[dependencies]
reqwest = "0.8.6"
failure = "0.1.1"

# I'll mention the remaining dependencies later
serde = "1.0.0"
serde_derive = "1.0.0"
serde_json = "1.0.0"
csv = "1.0.0"
sbr = "0.4.0"
rand = "0.5.4"
```


### main.rs {#main-dot-rs}

<a id="org02a6c2c"></a>
```rust
extern crate reqwest;
extern crate failure;

// Importing this allows us to autoderive
// the serialization traits.
#[macro_use]
extern crate serde_derive;

// This is where we get the serde traits from.
extern crate serde;

// An implementation of the serde encoders/decoders
// to and from a JSON. We'll need
// these later.
extern crate serde_json;

#[derive(Debug, Serialize, Deserialize)]
struct WishlistEntry {
    user_id: usize,
    book_id: usize,
}


// Need to import a couple of things from
// the standard library
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Download file from `url` and save it to `destination`.
fn download(url: &str, destination: &Path)
            -> Result<(), failure::Error> {

    // Don't do anything if we already have the file.
    if destination.exists() {
        return Ok(())
    }

    // Otherwise, create a new file.

    // Because each of the following operations
    // can fail (returns a result type), we follow
    // them with the `?` operator. If the result
    // is an error, it will exit from the function
    // early, propagating the error upwards; if
    // the operation completed successfully, we get
    // the result instead.
    let file = File::create(destination)?;
    let mut writer = BufWriter::new(file);

    let mut response = reqwest::get(url)?;
    response.copy_to(&mut writer)?;

    Ok(())
}
extern crate csv;

/// Deserialize from file at `path` into a vector of
/// `WishlistEntry`.
fn deserialize_ratings(path: &Path)
               -> Result<Vec<WishlistEntry>, failure::Error> {

    let mut reader = csv::Reader::from_path(path)?;

    // We specify the type of the deserialized entity
    // via a type annotation. Otherwise, the compiler has
    // no way of knowing what sort of thing we want to
    // deserialize!
    // We also do a further trick where instead of deserializing
    // into a vector of results, we deserialize into a result with
    // a vector.
    let entries: Vec<WishlistEntry> = reader.deserialize()
        .collect::<Result<Vec<_>, _>>()?;

    Ok(entries)
}
#[derive(Debug, Deserialize, Serialize)]
struct Book {
    book_id: usize,
    title: String
}

// We'll use the stdlib hashmap for the mapping.
use std::collections::HashMap;

/// Deserialize from file at `path` into the book
/// mappings.
fn deserialize_books(path: &Path)
   -> Result<(HashMap<usize, String>,
              HashMap<String, usize>), failure::Error> {

    let mut reader = csv::Reader::from_path(path)?;

    let entries: Vec<Book> = reader.deserialize::<Book>()
        .collect::<Result<Vec<_>, _>>()?;

    // We can simply iterate over the entries and collect
    // them into a different data structure. This is not
    // the most efficient solution but it will do for now.
    let id_to_title: HashMap<usize, String> = entries
        .iter()
        .map(|book| (book.book_id, book.title.clone()))
        .collect();
    let title_to_id: HashMap<String, usize> = entries
        .iter()
        .map(|book| (book.title.clone(), book.book_id))
        .collect();

    Ok((id_to_title, title_to_id))
}
extern crate sbr;

use sbr::models::ewma::{Hyperparameters, ImplicitEWMAModel};
use sbr::models::{Loss, Optimizer};

fn build_model(num_items: usize) -> ImplicitEWMAModel {
    let hyperparameters = Hyperparameters::new(num_items, 128)
        .embedding_dim(32)
        .learning_rate(0.16)
        .l2_penalty(0.0004)
        .loss(Loss::WARP)
        .optimizer(Optimizer::Adagrad)
        .num_epochs(10)
        .num_threads(1);

    hyperparameters.build()
}
use sbr::data::{Interaction, Interactions};

fn build_interactions(data: &[WishlistEntry]) -> Interactions {
    // If the collection is empty, `max` doesn't exist. This
    // is why we get an Option back, which we then unwrap.
    let num_users = data
        .iter()
        .map(|x| x.user_id)
        .max()
        .unwrap() + 1;
    let num_items = data
        .iter()
        .map(|x| x.book_id)
        .max()
        .unwrap() + 1;

    let mut interactions = Interactions::new(num_users,
                                             num_items);

    // There are no timestamps in the interaction data, but
    // we make use of the fact that they are sorted by time.
    for (idx, datum) in data.iter().enumerate() {
        interactions.push(
            Interaction::new(datum.user_id,
                             datum.book_id,
                             idx)
        );
    }

    interactions
}
// We need to import the rand crate.
extern crate rand;
use rand::SeedableRng;

// We perform a split where the train and test
// sets are disjoint on the user dimension: no
// single user is in both.
use sbr::data::user_based_split;
use sbr::OnlineRankingModel;

use sbr::evaluation::mrr_score;

/// Fit the model.
///
/// If successful, return the MRR on the test set.
/// Otherwise, return an error.
fn fit(model: &mut ImplicitEWMAModel,
       data: &Interactions)
       -> Result<f32, failure::Error> {

    // Use a fixed seed for repeatable results.
    let mut rng = rand::XorShiftRng::from_seed([42; 16]);

    let (train, test) = user_based_split(data,
                                         &mut rng,
                                         0.2);

    model.fit(&train.to_compressed())?;

    let mrr = mrr_score(model, &test.to_compressed())?;

    Ok(mrr)
}

fn serialize_model(model: &ImplicitEWMAModel,
                   path: &Path) -> Result<(), failure::Error> {

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    Ok(serde_json::to_writer(&mut writer, model)?)
}
/// Download training data and build a model.
///
/// We'll use this function to power the `fit`
/// subcommand of our command line tool.
fn main_build() {

    let ratings_path = Path::new("ratings.csv");
    let books_path = Path::new("books.csv");
    let model_path = Path::new("model.json");

    // Exit early if we already have a model.
    if model_path.exists() {
        println!("Model already fitted.");
        return ();
    }

    download_data(ratings_path, books_path);

    let ratings = deserialize_ratings(ratings_path).unwrap();
    let (id_to_title,
         title_to_id) = deserialize_books(books_path).unwrap();

    println!("Deserialized {} ratings.", ratings.len());
    println!("Deserialized {} books.", id_to_title.len());

    let interactions = build_interactions(&ratings);
    let mut model = build_model(interactions.num_items());

    println!("Fitting...");
    let mrr = fit(&mut model, &interactions)
        .expect("Unable to fit model.");
    println!("Fit model with MRR of {:.2}", mrr);

    serialize_model(&model, &model_path)
        .expect("Unable to serialize model.");
}
use std::io::BufReader;

fn deserialize_model() -> Result<ImplicitEWMAModel,
                                 failure::Error> {

    let file = File::open("model.json")?;
    let reader = BufReader::new(file);

    let model = serde_json::from_reader(reader)?;

    Ok(model)
}
fn predict(input_titles: &[String],
           model: &ImplicitEWMAModel)
           -> Result<Vec<String>, failure::Error> {
    let (id_to_title,
         title_to_id) = deserialize_books(
        &Path::new("books.csv")
    ).unwrap();

    // Let's first check if the inputs are valid.
    for title in input_titles {
        if !title_to_id.contains_key(title) {
            println!("No such title, ignoring: {}", title);
        }
    }

    // Map the titles to indices.
    let input_indices: Vec<_> = input_titles
        .iter()
        .filter_map(|title| title_to_id.get(title))
        .cloned()
        .collect();
    let indices_to_score: Vec<usize> =
        (0..id_to_title.len()).collect();

    // Get the user representation.
    let user = model.user_representation(&input_indices)?;
    // Get the actual predictions.
    let predictions = model.predict(&user, &indices_to_score)?;

    // We implement argsort by zipping item indices
    // with their scores into tuples...
    let mut predictions: Vec<_>
        = indices_to_score.iter()
        .zip(predictions)
        .map(|(idx, score)| (idx, score))
        .collect();

    // ...and sorting the result in descending order.
    // This is a little tricky for floats are they
    // are not always comparable (they could be NaN or Inf),
    // so we use partial sorting and fail the program
    // if non-finite values are encountered.
    predictions
        .sort_by(|(_, score_a), (_, score_b)|
                 score_b.partial_cmp(score_a)
                 .unwrap());

    // Finally, we get the names for the top 10 items.
    Ok((&predictions[..10])
       .iter()
       .map(|(idx, _)| id_to_title.get(idx).unwrap())
       .cloned()
       .collect())
}

/// Download ratings and metadata both.
fn download_data(ratings_path: &Path, books_path: &Path) {
    let ratings_url = "https://github.com/zygmuntz/\
                       goodbooks-10k/raw/master/ratings.csv";
    let books_url = "https://github.com/zygmuntz/\
                     goodbooks-10k/raw/master/books.csv";

    download(&ratings_url,
             ratings_path).expect("Could not download ratings");
    download(&books_url,
             books_path).expect("Could not download metadata");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        println!("First argument must be \
                  one of 'fit' or 'predict'.");
        return ();
    }

    // We need to convert a `String` into a
    // `&str` here. This is one of the few
    // cases where Rust's ergonomics still
    // have some way to go.
    match &args[0][..] {
        "fit" => main_build(),
        "predict" => {
            let model = deserialize_model()
                .expect("Unable to deserialize model.");
            let predictions = predict(&args[1..], &model)
                .expect("Unable to get predictions");
            println!("Predictions:");
            for prediction in predictions {
                println!("    {}", prediction);
            }
        },
        _ => println!("First argument must be \
                       one of 'fit' or 'predict'."),
    }
}
```
