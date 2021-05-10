#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright (c) 2021-2060 Curtis G. Northcutt
# This file is part of cgnorthcutt/label-errors.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cgnorthcutt/label-errors is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

# This agreement applies to this version and all previous versions of
# cgnorthcutt/label-errors.

"""
This tutorial provides reproducible code to find the label errors for 8 of
the 10 datasets, using only the pyx (predicted probs), pred (predicted labels),
and test label files, all of which are included in cgnorthcutt/label-errors.

In this tutorial, we exclude quickdraw because the pyx file is 33GB and might
cause trouble on some machines. We also exclude caltech-256 because we used a
low capacity model and its not reflective of recent performance.

This tutorial demonstrates that our results are reproducible and shows how we
find all label errors on labelerrors.com (prior to human validation on mTurk).
"""

import cleanlab
import numpy as np
import json
from util import get_file_size
# To view the text data from labelerrors.com, we need:
from urllib.request import urlopen
# To view the image data from labelerrors.com, we need:
from skimage import io
from matplotlib import pyplot as plt

# Remove axes since we're plotting images, not graphs
rc = {"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
      "xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False}
plt.rcParams.update(rc)


# In[2]:


cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
cifar100_classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
caltech256 = ["ak47", "american-flag", "backpack", "baseball-bat", "baseball-glove", "basketball-hoop", "bat", "bathtub", "bear", "beer-mug", "billiards", "binoculars", "birdbath", "blimp", "bonsai-101", "boom-box", "bowling-ball", "bowling-pin", "boxing-glove", "brain-101", "breadmaker", "buddha-101", "bulldozer", "butterfly", "cactus", "cake", "calculator", "camel", "cannon", "canoe", "car-tire", "cartman", "cd", "centipede", "cereal-box", "chandelier-101", "chess-board", "chimp", "chopsticks", "cockroach", "coffee-mug", "coffin", "coin", "comet", "computer-keyboard", "computer-monitor", "computer-mouse", "conch", "cormorant", "covered-wagon", "cowboy-hat", "crab-101", "desk-globe", "diamond-ring", "dice", "dog", "dolphin-101", "doorknob", "drinking-straw", "duck", "dumb-bell", "eiffel-tower", "electric-guitar-101", "elephant-101", "elk", "ewer-101", "eyeglasses", "fern", "fighter-jet", "fire-extinguisher", "fire-hydrant", "fire-truck", "fireworks", "flashlight", "floppy-disk", "football-helmet", "french-horn", "fried-egg", "frisbee", "frog", "frying-pan", "galaxy", "gas-pump", "giraffe", "goat", "golden-gate-bridge", "goldfish", "golf-ball", "goose", "gorilla", "grand-piano-101", "grapes", "grasshopper", "guitar-pick", "hamburger", "hammock", "harmonica", "harp", "harpsichord", "hawksbill-101", "head-phones", "helicopter-101", "hibiscus", "homer-simpson", "horse", "horseshoe-crab", "hot-air-balloon", "hot-dog", "hot-tub", "hourglass", "house-fly", "human-skeleton", "hummingbird", "ibis-101", "ice-cream-cone", "iguana", "ipod", "iris", "jesus-christ", "joy-stick", "kangaroo-101", "kayak", "ketch-101", "killer-whale", "knife", "ladder", "laptop-101", "lathe", "leopards-101", "license-plate", "lightbulb", "light-house", "lightning", "llama-101", "mailbox", "mandolin", "mars", "mattress", "megaphone", "menorah-101", "microscope", "microwave", "minaret", "minotaur", "motorbikes-101", "mountain-bike", "mushroom", "mussels", "necktie", "octopus", "ostrich", "owl", "palm-pilot", "palm-tree", "paperclip", "paper-shredder", "pci-card", "penguin", "people", "pez-dispenser", "photocopier", "picnic-table", "playing-card", "porcupine", "pram", "praying-mantis", "pyramid", "raccoon", "radio-telescope", "rainbow", "refrigerator", "revolver-101", "rifle", "rotary-phone", "roulette-wheel", "saddle", "saturn", "school-bus", "scorpion-101", "screwdriver", "segway", "self-propelled-lawn-mower", "sextant", "sheet-music", "skateboard", "skunk", "skyscraper", "smokestack", "snail", "snake", "sneaker", "snowmobile", "soccer-ball", "socks", "soda-can", "spaghetti", "speed-boat", "spider", "spoon", "stained-glass", "starfish-101", "steering-wheel", "stirrups", "sunflower-101", "superman", "sushi", "swan", "swiss-army-knife", "sword", "syringe", "tambourine", "teapot", "teddy-bear", "teepee", "telephone-box", "tennis-ball", "tennis-court", "tennis-racket", "theodolite", "toaster", "tomato", "tombstone", "top-hat", "touring-bike", "tower-pisa", "traffic-light", "treadmill", "triceratops", "tricycle", "trilobite-101", "tripod", "t-shirt", "tuning-fork", "tweezer", "umbrella-101", "unicorn", "vcr", "video-projector", "washing-machine", "watch-101", "waterfall", "watermelon", "welding-mask", "wheelbarrow", "windmill", "wine-bottle", "xylophone", "yarmulke", "yo-yo", "zebra", "airplanes-101", "car-side-101", "faces-easy-101", "greyhound", "tennis-shoes", "toad", "clutter"]
news_groups = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
amazon_reviews = ['Negative', 'Neutral', 'Positive']
quickdraw = ["aircraft carrier", "airplane", "alarm clock", "ambulance", "angel", "animal migration", "ant", "anvil", "apple", "arm", "asparagus", "axe", "backpack", "banana", "bandage", "barn", "baseball", "baseball bat", "basket", "basketball", "bat", "bathtub", "beach", "bear", "beard", "bed", "bee", "belt", "bench", "bicycle", "binoculars", "bird", "birthday cake", "blackberry", "blueberry", "book", "boomerang", "bottlecap", "bowtie", "bracelet", "brain", "bread", "bridge", "broccoli", "broom", "bucket", "bulldozer", "bus", "bush", "butterfly", "cactus", "cake", "calculator", "calendar", "camel", "camera", "camouflage", "campfire", "candle", "cannon", "canoe", "car", "carrot", "castle", "cat", "ceiling fan", "cello", "cell phone", "chair", "chandelier", "church", "circle", "clarinet", "clock", "cloud", "coffee cup", "compass", "computer", "cookie", "cooler", "couch", "cow", "crab", "crayon", "crocodile", "crown", "cruise ship", "cup", "diamond", "dishwasher", "diving board", "dog", "dolphin", "donut", "door", "dragon", "dresser", "drill", "drums", "duck", "dumbbell", "ear", "elbow", "elephant", "envelope", "eraser", "eye", "eyeglasses", "face", "fan", "feather", "fence", "finger", "fire hydrant", "fireplace", "firetruck", "fish", "flamingo", "flashlight", "flip flops", "floor lamp", "flower", "flying saucer", "foot", "fork", "frog", "frying pan", "garden", "garden hose", "giraffe", "goatee", "golf club", "grapes", "grass", "guitar", "hamburger", "hammer", "hand", "harp", "hat", "headphones", "hedgehog", "helicopter", "helmet", "hexagon", "hockey puck", "hockey stick", "horse", "hospital", "hot air balloon", "hot dog", "hot tub", "hourglass", "house", "house plant", "hurricane", "ice cream", "jacket", "jail", "kangaroo", "key", "keyboard", "knee", "knife", "ladder", "lantern", "laptop", "leaf", "leg", "light bulb", "lighter", "lighthouse", "lightning", "line", "lion", "lipstick", "lobster", "lollipop", "mailbox", "map", "marker", "matches", "megaphone", "mermaid", "microphone", "microwave", "monkey", "moon", "mosquito", "motorbike", "mountain", "mouse", "moustache", "mouth", "mug", "mushroom", "nail", "necklace", "nose", "ocean", "octagon", "octopus", "onion", "oven", "owl", "paintbrush", "paint can", "palm tree", "panda", "pants", "paper clip", "parachute", "parrot", "passport", "peanut", "pear", "peas", "pencil", "penguin", "piano", "pickup truck", "picture frame", "pig", "pillow", "pineapple", "pizza", "pliers", "police car", "pond", "pool", "popsicle", "postcard", "potato", "power outlet", "purse", "rabbit", "raccoon", "radio", "rain", "rainbow", "rake", "remote control", "rhinoceros", "rifle", "river", "roller coaster", "rollerskates", "sailboat", "sandwich", "saw", "saxophone", "school bus", "scissors", "scorpion", "screwdriver", "sea turtle", "see saw", "shark", "sheep", "shoe", "shorts", "shovel", "sink", "skateboard", "skull", "skyscraper", "sleeping bag", "smiley face", "snail", "snake", "snorkel", "snowflake", "snowman", "soccer ball", "sock", "speedboat", "spider", "spoon", "spreadsheet", "square", "squiggle", "squirrel", "stairs", "star", "steak", "stereo", "stethoscope", "stitches", "stop sign", "stove", "strawberry", "streetlight", "string bean", "submarine", "suitcase", "sun", "swan", "sweater", "swing set", "sword", "syringe", "table", "teapot", "teddy-bear", "telephone", "television", "tennis racquet", "tent", "Eiffel Tower", "Great Wall of China", "The Mona Lisa", "tiger", "toaster", "toe", "toilet", "tooth", "toothbrush", "toothpaste", "tornado", "tractor", "traffic light", "train", "tree", "triangle", "trombone", "truck", "trumpet", "t-shirt", "umbrella", "underwear", "van", "vase", "violin", "washing machine", "watermelon", "waterslide", "whale", "wheel", "windmill", "wine bottle", "wine glass", "wristwatch", "yoga", "zebra", "zigzag"]
audioset = ["Music", "Speech", "Vehicle", "Musical instrument", "Plucked string instrument", "Singing", "Car", "Animal", "Outside, rural or natural", "Violin, fiddle", "Bird", "Drum", "Engine", "Narration, monologue", "Drum kit", "Acoustic guitar", "Dog", "Child speech, kid speaking", "Bass drum", "Rail transport", "Motor vehicle (road)", "Water", "Female speech, woman speaking", "Siren", "Railroad car, train wagon", "Tools", "Silence", "Snare drum", "Wind", "Bird vocalization, bird call, bird song", "Fowl", "Wind instrument, woodwind instrument", "Emergency vehicle", "Laughter", "Chirp, tweet", "Rapping", "Cheering", "Gunshot, gunfire", "Radio", "Cat", "Hi-hat", "Helicopter", "Fireworks", "Stream", "Bark", "Baby cry, infant cry", "Snoring", "Train horn", "Double bass", "Explosion", "Crowing, cock-a-doodle-doo", "Bleat", "Computer keyboard", "Civil defense siren", "Bee, wasp, etc.", "Bell", "Chainsaw", "Oink", "Tick", "Tabla", "Liquid", "Traffic noise, roadway noise", "Beep, bleep", "Frying (food)", "Whack, thwack", "Sink (filling or washing)", "Burping, eructation", "Fart", "Sneeze", "Aircraft engine", "Arrow", "Giggle", "Hiccup", "Cough", "Cricket", "Sawing", "Tambourine", "Pump (liquid)", "Squeak", "Male speech, man speaking", "Keyboard (musical)", "Pigeon, dove", "Motorboat, speedboat", "Female singing", "Brass instrument", "Motorcycle", "Choir", "Race car, auto racing", "Chicken, rooster", "Idling", "Sampler", "Ukulele", "Synthesizer", "Cymbal", "Spray", "Accordion", "Scratching (performance technique)", "Child singing", "Cluck", "Water tap, faucet", "Applause", "Toilet flush", "Whistling", "Vacuum cleaner", "Meow", "Chatter", "Whoop", "Sewing machine", "Bagpipes", "Subway, metro, underground", "Walk, footsteps", "Whispering", "Crying, sobbing", "Thunder", "Didgeridoo", "Church bell", "Ringtone", "Buzzer", "Splash, splatter", "Fire alarm", "Chime", "Babbling", "Glass", "Chewing, mastication", "Microwave oven", "Air horn, truck horn", "Growling", "Telephone bell ringing", "Moo", "Change ringing (campanology)", "Hands", "Camera", "Pour", "Croak", "Pant", "Finger snapping", "Gargling", "Inside, small room", "Outside, urban or manmade", "Truck", "Bowed string instrument", "Medium engine (mid frequency)", "Marimba, xylophone", "Aircraft", "Cello", "Flute", "Glockenspiel", "Power tool", "Fixed-wing aircraft, airplane", "Waves, surf", "Duck", "Clarinet", "Goat", "Honk", "Skidding", "Hammond organ", "Electronic organ", "Thunderstorm", "Steelpan", "Slap, smack", "Battle cry", "Percussion", "Trombone", "Banjo", "Mandolin", "Guitar", "Strum", "Boat, Water vehicle", "Accelerating, revving, vroom", "Electric guitar", "Orchestra", "Wind noise (microphone)", "Effects unit", "Livestock, farm animals, working animals", "Police car (siren)", "Rain", "Printer", "Drum machine", "Fire engine, fire truck (siren)", "Insect", "Skateboard", "Coo", "Conversation", "Typing", "Harp", "Thump, thud", "Mechanisms", "Canidae, dogs, wolves", "Chuckle, chortle", "Rub", "Boom", "Hubbub, speech noise, speech babble", "Telephone", "Blender", "Whimper", "Screaming", "Wild animals", "Pig", "Artillery fire", "Electric shaver, electric razor", "Baby laughter", "Crow", "Howl", "Breathing", "Cattle, bovinae", "Roaring cats (lions, tigers)", "Clapping", "Alarm", "Chink, clink", "Ding", "Toot", "Clock", "Children shouting", "Fill (with liquid)", "Purr", "Rumble", "Boing", "Breaking", "Light engine (high frequency)", "Cash register", "Bicycle bell", "Inside, large room or hall", "Domestic animals, pets", "Bass guitar", "Electric piano", "Trumpet", "Horse", "Mallet percussion", "Organ", "Bicycle", "Rain on surface", "Quack", "Drill", "Machine gun", "Lawn mower", "Smash, crash", "Trickle, dribble", "Frog", "Writing", "Steam whistle", "Groan", "Hammer", "Doorbell", "Shofar", "Cowbell", "Wail, moan", "Bouncing", "Distortion", "Vibraphone", "Air brake", "Field recording", "Piano", "Male singing", "Bus", "Wood", "Tap", "Ocean", "Door", "Vibration", "Television", "Harmonica", "Basketball bounce", "Clickety-clack", "Dishes, pots, and pans", "Crumpling, crinkling", "Sitar", "Tire squeal", "Fly, housefly", "Sizzle", "Slosh", "Engine starting", "Mechanical fan", "Stir", "Children playing", "Ping", "Owl", "Alarm clock", "Car alarm", "Telephone dialing, DTMF", "Sine wave", "Thunk", "Coin (dropping)", "Crunch", "Zipper (clothing)", "Mosquito", "Shuffling cards", "Pulleys", "Toothbrush", "Crowd", "Saxophone", "Rowboat, canoe, kayak", "Steam", "Ambulance (siren)", "Goose", "Crackle", "Fire", "Turkey", "Heart sounds, heartbeat", "Singing bowl", "Reverberation", "Clicking", "Jet engine", "Rodents, rats, mice", "Typewriter", "Caw", "Knock", "Ice cream truck, ice cream van", "Stomach rumble", "French horn", "Roar", "Theremin", "Pulse", "Train", "Run", "Vehicle horn, car horn, honking", "Clip-clop", "Sheep", "Whoosh, swoosh, swish", "Timpani", "Throbbing", "Firecracker", "Belly laugh", "Train whistle", "Whistle", "Whip", "Gush", "Biting", "Scissors", "Clang", "Single-lens reflex camera", "Chorus effect", "Inside, public space", "Steel guitar, slide guitar", "Waterfall", "Hum", "Raindrop", "Propeller, airscrew", "Filing (rasp)", "Reversing beeps", "Shatter", "Sanding", "Wheeze", "Hoot", "Bow-wow", "Car passing by", "Tick-tock", "Hiss", "Snicker", "Whimper (dog)", "Shout", "Echo", "Rattle", "Sliding door", "Gobble", "Plop", "Yell", "Drip", "Neigh, whinny", "Bellow", "Keys jangling", "Ding-dong", "Buzz", "Scratch", "Rattle (instrument)", "Hair dryer", "Dial tone", "Tearing", "Bang", "Noise", "Bird flight, flapping wings", "Grunt", "Jackhammer", "Drawer open or close", "Whir", "Tuning fork", "Squawk", "Jingle bell", "Smoke detector, smoke alarm", "Train wheels squealing", "Caterwaul", "Mouse", "Crack", "Whale vocalization", "Squeal", "Zither", "Rimshot", "Drum roll", "Burst, pop", "Wood block", "Harpsichord", "White noise", "Bathtub (filling or washing)", "Snake", "Environmental noise", "String section", "Cacophony", "Maraca", "Snort", "Yodeling", "Electric toothbrush", "Cupboard open or close", "Sound effect", "Tapping (guitar technique)", "Ship", "Sniff", "Pink noise", "Tubular bells", "Gong", "Flap", "Throat clearing", "Sigh", "Busy signal", "Zing", "Sidetone", "Crushing", "Yip", "Gurgling", "Jingle, tinkle", "Boiling", "Mains hum", "Humming", "Sonar", "Gasp", "Power windows, electric windows", "Splinter", "Heart murmur", "Air conditioning", "Pizzicato", "Ratchet, pawl", "Chirp tone", "Heavy engine (low frequency)", "Rustling leaves", "Speech synthesizer", "Rustle", "Clatter", "Slam", "Eruption", "Cap gun", "Synthetic singing", "Shuffle", "Wind chime", "Chop", "Scrape", "Squish", "Foghorn", "Dental drill, dentist's drill", "Harmonic", "Static", "Sailboat, sailing ship", "Cutlery, silverware", "Gears", "Chopping (food)", "Creak", "Fusillade", "Roll", "Electronic tuner", "Patter", "Electronic music", "Dubstep", "Techno", "Rock and roll", "Pop music", "Rock music", "Hip hop music", "Classical music", "Soundtrack music", "House music", "Heavy metal", "Exciting music", "Country", "Electronica", "Rhythm and blues", "Background music", "Dance music", "Jazz", "Mantra", "Blues", "Trance music", "Electronic dance music", "Theme music", "Gospel music", "Music of Latin America", "Disco", "Tender music", "Punk rock", "Funk", "Music of Asia", "Drum and bass", "Vocal music", "Progressive rock", "Music for children", "Video game music", "Lullaby", "Reggae", "New-age music", "Christian music", "Independent music", "Soul music", "Music of Africa", "Ambient music", "Bluegrass", "Afrobeat", "Salsa music", "Music of Bollywood", "Beatboxing", "Flamenco", "Psychedelic rock", "Opera", "Folk music", "Christmas music", "Middle Eastern music", "Grunge", "Song", "A capella", "Sad music", "Traditional music", "Scary music", "Ska", "Chant", "Carnatic music", "Swing music", "Happy music", "Jingle (music)", "Funny music", "Angry music", "Wedding music", "Engine knocking"]
imdb = ["Negative", "Positive"]
mnist = ["0", "1" ,"2", "3", "4", "5", "6", "7", "8", "9"]
imagenet = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "cock", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peacock", "quail", "partridge", "grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern", "crane (bird)", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniels", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland", "Pyrenean Mountain Dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "Griffon Bruxellois", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog", "grey wolf", "Alaskan tundra wolf", "red wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket", "stick insect", "cockroach", "mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral", "ringlet", "monarch butterfly", "small white", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram", "bighorn sheep", "Alpine ibex", "hartebeest", "impala", "gazelle", "dromedary", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek", "eel", "coho salmon", "rock beauty", "clownfish", "sturgeon", "garfish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "waste container", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military cap", "beer bottle", "beer glass", "bell-cot", "bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "bow", "bow tie", "brass", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "chest", "chiffonier", "chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "coil", "combination lock", "computer keyboard", "confectionery store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "crane (machine)", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire engine", "fire screen sheet", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "grille", "grocery store", "guillotine", "barrette", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "jack-o'-lantern", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "pulled rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "paper knife", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "speaker", "loupe", "sawmill", "magnetic compass", "mail bag", "mailbox", "tights", "tank suit", "manhole cover", "maraca", "marimba", "mask", "match", "maypole", "maze", "measuring cup", "medicine chest", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "Model T", "modem", "monastery", "monitor", "moped", "mortar", "square academic cap", "mosque", "mosquito net", "scooter", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "nail", "neck brace", "necklace", "nipple", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "packet", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "passenger car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "pitcher", "hand plane", "planetarium", "plastic bag", "plate rack", "plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "billiard table", "soda bottle", "pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "projectile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler", "running shoe", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT screen", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swimsuit", "swing", "switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vault", "velvet", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "wig", "window screen", "window shade", "Windsor tie", "wine bottle", "wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "yawl", "yurt", "website", "comic book", "crossword", "traffic sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "ice pop", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potato", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "custard apple", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "cup", "eggnog", "alp", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "shoal", "seashore", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star", "hen-of-the-woods", "bolete", "ear", "toilet paper"]

ALL_CLASSES = {
    'mnist_test_set': mnist,
    'cifar10_test_set': cifar10_classes,
    'cifar100_test_set': cifar100_classes,
    'caltech256': caltech256,
    'imagenet_val_set': imagenet,
    'quickdraw': quickdraw,
    '20news_test_set': news_groups,
    'imdb_test_set': imdb,
    'amazon': amazon_reviews,
    'audioset_eval_set': audioset,
}


# In[3]:


datasets = [
    ('imagenet_val_set', 'image'),
    ('mnist_test_set', 'image'),
    ('cifar10_test_set', 'image'),
    ('cifar100_test_set', 'image'),
    ('imdb_test_set', 'text'),
    ('20news_test_set', 'text'),
    ('amazon', 'text'),
    ('audioset_eval_set', 'audio'),
]


# In[4]:


with open("../resources/audioset_eval_set_index_to_youtube_id.json", 'r') as rf:
    AUDIOSET_INDEX_TO_YOUTUBE = json.load(rf)
with open("../resources/imdb_test_set_index_to_filename.json", 'r') as rf:
    IMDB_INDEX_TO_FILENAME = json.load(rf)
with open("../resources/imagenet_val_set_index_to_filepath.json", 'r') as rf:
    IMAGENET_INDEX_TO_FILEPATH = json.load(rf)


# In[5]:


for (dataset, modality) in datasets:
    title = 'Dataset: ' + dataset.capitalize()
    print('='*len(title), title, '='*len(title), sep='\n')
    
    # Get the cross-validated predicted probabilities on the test set.
    if dataset == 'amazon' or dataset == 'imagenet_val_set':
        n_parts = 3 if dataset == 'amazon' else 4
        pyx_fn = '../cross_validated_predicted_probabilities/'              '{}_pyx.part{}_of_{}.npy'
        parts = [np.load(pyx_fn.format(dataset, i + 1, n_parts)) for i in range(n_parts)]
        pyx = np.vstack(parts)
    else:
        pyx = np.load('../cross_validated_predicted_probabilities/'             '{}_pyx.npy'.format(dataset), allow_pickle=True)
    # Get the cross-validated predictions (argmax of pyx) on the test set.
    pred = np.load('../cross_validated_predicted_labels/'
        '{}_pyx_argmax_predicted_labels.npy'.format(dataset), allow_pickle=True)
    # Get the test set labels
    test_labels = np.load('../original_test_labels/'
        '{}_original_labels.npy'.format(dataset), allow_pickle=True)
    
    # Find label error indices using cleanlab in one line of code.
    print('Finding label errors using cleanlab for {:,} examples and {} classes...'.format(*pyx.shape))
    label_error_indices = cleanlab.pruning.get_noise_indices(
        s=test_labels, # np.asarray([z[0] for z in y_test]),
        psx=pyx,
        prune_method='both',
        multi_label=True if dataset == 'audioset_eval_set' else False,
        sorted_index_method='normalized_margin',
    )
    
    # Grab a label error found with cleanlab
    err_id = label_error_indices[0]
    
    # Custom code to visualize each label error from each dataset
    dname = dataset.split('_')[0]  # Get dataset name
    url_base = "https://labelerrors.com/static/{}/{}".format(dname, err_id)
    if modality == 'image':
        if dataset == 'imagenet_val_set':
            image_path = IMAGENET_INDEX_TO_FILEPATH[err_id]
            url = url_base.replace(str(err_id), image_path)
        else:
            url = url_base + ".png"
        image = io.imread(url)  # read image data from a url
        plt.imshow(image, interpolation='nearest', aspect='auto', cmap='gray')
        plt.show()
    elif modality == 'text':
        if dataset == 'amazon':
            # There are 400,000+ amazon reviews errors -- we only check a small
            # fraction on labelerrors.com, so choose one that's on the website.
            err_id = 8864504
            assert err_id in label_error_indices  # Did we find this error?
            url = "https://labelerrors.com/static/{}/{}.txt".format(dname, err_id)
        elif dataset == 'imdb_test_set':
            imdb_fn = IMDB_INDEX_TO_FILENAME[err_id]  
            url = "https://labelerrors.com/static/{}/test/{}".format(dname, imdb_fn)
        else:
            url = url_base + ".txt"
        text = urlopen(url).read().decode("utf-8")  # read raw text from a url
        print('\n{} Text Example (ID: {}):\n{}\n'.format(
            dataset.capitalize(), err_id, text))
    elif modality == 'audio':  # dataset == 'audioset_eval_set'
        # Because AudioSet is multi-label, we only look at examples where the 
        # predictions have no overlap with the labels to avoid overcounting.
        label_error_indices = [z for z in label_error_indices                 if set(pred[z]).intersection(test_labels[z]) == set()]
        err_id = label_error_indices[1]
        youtube_id = AUDIOSET_INDEX_TO_YOUTUBE[err_id]
        url = 'https://youtu.be/{}'.format(youtube_id)
    # Map label indices to class names
    if dataset == 'audioset_eval_set':  # multi-label    
        given_label = [ALL_CLASSES[dataset][z] for z in test_labels[err_id]]
        pred_label = [ALL_CLASSES[dataset][z] for z in pred[err_id]]
    else:  # single-label
        given_label = ALL_CLASSES[dataset][test_labels[err_id]]
        pred_label = ALL_CLASSES[dataset][pred[err_id]]
    print(' * {} Given Label:'.format(dataset.capitalize()), given_label)
    print(' * We Guess (argmax prediction):', pred_label)
    print(' * Label Error Found: {}\n'.format(url))

