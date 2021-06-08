# Mechanical Turk Study Results

Each file is JSON-encoded, with varying schemas. The `id` of each element is
the index into the corresponding files in predicted labels / predicted
probabilities / original labels. Each data point was assigned to five
Mechanical Turk workers.

## Details

**Image datasets**: MNIST, CIFAR-10, CIFAR-100, ImageNet, Caltech-256, and
QuickDraw results are all in the same format. For each element, the `mturk`
object indicates how many workers thought the image contained the `given`
label, the `guessed` label, `neither` label, or `both` labels.

**Text datasets**: 20news has a `mturk` object with the same schema as the
image datasets. The IMDB `mturk` object indicates whether the worker selected
the `given` label, `guessed` label, `neutral`, or `off-topic`. The Amazon
`mturk` object object indicates whether the review was `positive`, `negative`,
`neutral`, or `off-topic`.

**Audio datasets**: AudioSet is a multi-label dataset; the AudioSet results
file lists all the `given_original_labels`, `our_guessed_labels`, and the
Mechanical Turk results, indicating how many workers voted for each of the
labels in the set of given/guessed labels.
