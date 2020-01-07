# 1. Charade

![logo](./charade.png)

A server for multilanguage, composable NLP API in Python.

<!-- TOC -->

- [Charade](#charade)
  - [Philosophy](#philosophy)
  - [What Charade is and is not](#what-charade-is-and-is-not)
  - [Installing](#installing)
    - [Using Pipenv (recommended)](#using-pipenv-recommended)
      - [Common errors](#common-errors)
    - [Using Conda and Pip](#using-conda-and-pip)
  - [Running](#running)
  - [Docker running](#docker-running)
  - [Endpoints](#endpoints)
  - [Architecture](#architecture)
  - [Requests](#requests)
    - [Resumable requests](#resumable-requests)
  - [Describing services](#describing-services)
  - [Services](#services)
    - [Parsing](#parsing)
    - [NER](#ner)
    - [Date extraction](#date-extraction)
    - [Codes extraction](#codes-extraction)
    - [Fiscal codes](#fiscal-codes)
    - [Extractive summarization](#extractive-summarization)
    - [Keyword extraction](#keyword-extraction)
    - [Sentiment detection](#sentiment-detection)
    - [Names](#names)
    - [Topic modeling](#topic-modeling)
    - [Classification](#classification)
  - [How to create a new service](#how-to-create-a-new-service)
  - [Testing](#testing)
  - [Style guide](#style-guide)
  - [Organization](#organization)

<!-- /TOC -->

## 1.1. Philosophy

Charade was born as a container where multiple independent natural language
services can coexist and interact with each other. In order to develop on
Charade, it may be useful to understand the reasons behind its implementation.

* multiple analyses can be run over a single text - for instance named entity
  recognition and sentiment detection - so a request from a user should be
  able to specify what kind of tasks should be performed on the provided text
* to avoid repeting work and ensure consistency, one task may be dependent on
  another: for instance, if both the NER and sentiment analysis rely on the same
  parsing stage, they will get to see the same tokens, something which would not
  be guaranteed if the two analyses performed tokenization internally
* a single task could have many coexisting implementations, so that a developer
  would be free to experiment with new models without having to interfere with
  existing ones. The user consuming the service could then be able to request
  a particular implementation of a task by specifying its name
* multiple implementations of a single task should offer a consistent
  interface, in order to ensure that clients or other downstream services can
  switch between them freely
* the server should not be restricted to a single (natural) language, and
  various implementations should be free to decide what languages to support
* developers implementing various models should be able to choose freely what
  technology to use, so various services can be implemented on top of NLTK,
  spaCy, pyTorch, TensorFlow, GenSim... Charade should make it easy to use any
  of these libraries to implement a particular model, without forcing other
  developers to adopt the same library
* one should be able to implement as many tasks and models as desired, while
  choosing at deploy time which one are supported by the server - i.e. the
  server should be composable from Lego pieces

Therefore, the process of deploying Charade servers works as follows. The
developers write various models to perform some tasks, possibly trying
competing implementations in parallel. Various kind of models are already
provided with Charade, but you should not shy from writing your own.

Once the models are ready, one writes an entry point script that actually
loads only the ones that will be used in production. At every point of the
process, one has available an API offering the existing models, and a user
interface to try them.

## 1.2. What Charade is and is not

Charade is a framework that helps teams experimenting with multiple approaches
to tackle some custom NLP task. It is meant to leverage existing NLP libraries,
such as NLTK or spaCy, and not to replace them. A team using Charade can develop
and evolve a suite of NLP capabilities - say NER, sentiment analysis and so on -
while maintaining the possibility to customize them on particular datasets, and
compose servers where only the relevant capabilities are deployed.

Charade is not itself a library for NLP tasks, although it provides some examples
of models developed using various libraries. It is not a ready-made component
either: while some of the models provided can be useful, we expect that teams
using Charade will develop and customize their own models. The provided ones
can serve as example, or can provide some capabilities in a larger deployment.

## 1.3. Installing

**NB** If you are on MacOS Mojave, make sure to have the XCode headers installed

```
xcode-select --install
open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
```

Also, OpenMP is required by PyTorch, on MacOS it can be installed by

```
brew install libomp
```

### 1.3.1. Using Pipenv (recommended)

Install Pipenv if needed (`pip install pipenv`). An introduction to Pipenv
can be found [here](https://realpython.com/pipenv-guide/).

Create a virtual environment related to this project by running `pipenv shell`
from inside the top directory in the project.

1. If you want to **develop Charade**, you can install dependencies with this command:

```
pipenv install --dev
```

If you also make the iPython kernel for Charade visible to other environments,
you can use

```
python -m ipykernel install --user --name="charade"
```

In this way, you can use any installation of Jupyter to launch the `charade`
kernel.

2. If instead you want to **try Charade** without developing, then run

```
pipenv install --ignore-pipfile
```

to install all dependencies.

3. In **both cases**, download the models for `spacy`, `allen-nlp` and `nltk` via

```
python -m spacy download en
python -m spacy download it
python -m spacy download de

python -m nltk.downloader averaged_perceptron_tagger
python -m nltk.downloader maxent_ne_chunker
python -m nltk.downloader words

mkdir -p models/allen/pretrained
wget https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz -O models/allen/pretrained/ner-model-2018.12.18.tar.gz
```

#### 1.3.1.1. Common errors

**NB** If you get an error that you don't have the right version of Pyhton,
you can manage that through PyEnv.  To install PyEnv, see
[the installation instructions](https://github.com/pyenv/pyenv#installation).
On MacOS just run `brew install pyenv`. After having install PyEnv, install
the required version of Pyhton, for instance `pyenv install 3.6.8`.

After this step, `pipenv` should detect the version of pyenv automatically.

### 1.3.2. Using Conda and Pip

If you **don't need to develop Charade** itself, you can create a virtual environment
in Conda by running something like `conda create -n charade python=3.6`, then
activate it with `source activate charade` (any other name will do). Then install
dependencies with Pip inside the environment:

```
pip install -r requirements.txt
```

Finally, update `spacy` models via

```
python -m spacy download en
python -m spacy download it
```

**NB** The `requirements.txt file` is autogenerated by Pipenv with the command
`pipenv lock --requirements > requirements.txt` - do not edit this file by hand.

## 1.4. Running

Just define the server in `src/main.py`, then run

```
python src/main.py
```

The existing `main.py` file only contains those models that do not require
a custom training step. The other models are commented. You can launch any of
the traning scripts - they are ready, but may be trained on toy datasets, so be
ready to adjust them to your needs - and then uncomment the resulting models in
the main script.

Once you have a running server, you can try some queries. An example query can be
sent using `examples/request.sh`. You can pass a parameter to select a particular
request, for instance

```bash
examples/request.sh reprise
```

You can see available examples with `ls examples`.

Also, there is a frontend available at `http://localhost:9000/app`.

## 1.5. Docker running

The docker can be built by using `scripts/build-docker.sh`. Then, to run the
docker container simply do

```bash
docker-compose up
```

**NB** Since both `uwsgi` and some services (e.g. pytorch) make use of multiple
threads, this can cause deadlocks. To avoid them, we need to run the uwsgi command
with the option `--lazy-apps` as specified in the `Dockerfile` (see
https://engineering.ticketea.com/uwsgi-preforking-lazy-apps/ for an explanation
of this mechanism).
Note that if the uwsgi option `--processes` is > 1, each worker will load the full
application and thus the server startup may require **a lot** of time and memory.
By employing multiple threads and a single process instead (e.g. `--processes 1 --threads 4`)
the server startup is fast enough.

## 1.6. Endpoints

A Charade server had just two endpoints:

* GET `/`: returns a JSON describing the available services
* POST `/`: post a request with a text and some services to be performed

## 1.7. Architecture

A Charade server is defined by instantiating and putting together various
services. Each service is defined by

* a **task**
* a **service name**
* optional **dependencies**
* an actual **implementation**.

**Tasks** are used to denote interchangeable services. For instance, there may
exist various NER models, possibly using different libraries and technologies.
In this case, we will define a `ner` task, with the only requirement that if
there are various implementations of `ner`, they need to abide to the same
interface.

**Names** are used to distinguish different implementations of the same task.
The task/name pair should identify a unique service. For instance, one could
have deployed `ner` services named `allen`, `nltk`, `pytorch-crf`, `pytorch-crf-2`.

**Dependencies** can be used to avoid repeating the same task over and over.
For instance, a `ner` implementation may (or may not) depend on some implementation
of the `parse` task, which takes care of tokenization. At runtime, the server
will ensure that the `parse` task is executed before `ner`.

The precise mechanism is as follows. The user request contains a field called
`tasks`, which contains the list of tasks to be executed on the given chunk of
text. For instance:

```json
  "tasks": [
    {"task": "parse", "name": "spacy"},
    {"task": "ner", "name": "allen"},
    {"task": "dates", "name": "misc"}
  ]
```

Tasks are executed in the order requested by the user. The objects returned by
the various tasks populate corresponding fields in a `response` dictionary. For
instance, for this request, the `response` object will have the shape

```json
{
  "parse": ...,
  "ner": ...,
  "dates": ...
}
```

Each service can look at the `request` object and the `response` object (the
part that has been populated so far). In this way, a service can look at the
output produced by other services that come before.

If a dependency for a service has not been requested explicitly by the user,
the server will choose *any* implementation of the dependency task and execute
it *before* the dependent task. For instance, say one has a `ner` service called
`custom` which depends on `parse`. If the user request contains

```json
  "tasks": [
    {"task": "ner", "name": "custom"},
    {"task": "dates", "name": "misc"}
  ]
```

then the server will choose *any* implementation of `parse` and perform it
before `ner`. This has two advantages:

* duplication is reduced, for instance the parsing and tokenization of the text
  can be done just once and many other services can consume it
* one has the guarantee that all services rely on the same tokenization, giving
  a better consistency.

**Implementations** are defined by writing a class that inherits from
`services.Service`. The methods to override are `Service.run(request, response)`
and `Service.describe()` (optional, but recommended).
The former has access to

* the user request
* the part of the response constructed so far

and has to return a dictionary containing the service output. This method can
raise `services.MissingLanguage` if the language of the request is not
supported in the given service. The class should load any needed model in its
constructor, to avoid reloading models for each request.

For instance, a trivial parser that just splits sentences on period and tokens
on whitespace may look like this:

```python
from services import Service

class SimpleParser(Service):
    def __init__(self):
        pass

    def run(self, request, response):
        text = request['text']
        debug = request.get('debug', False)
        result = []
        start = 0
        end = 0
        for sentence in text.split('\.'):
            tokens = []
            for token in sentence.split(' '):
                start = end + 1
                end += start + len(token)
                if debug:
                    tokens.append({
                        'text': token,
                        'start': start,
                        'end': end
                    })
                else:
                    tokens.append({
                        'start': start,
                        'end': end
                    })
            result.append(tokens)
        return result
```

## 1.8. Requests

The user requests have the following fields:

* `text`: required, the text to be analyzed
* `debug`: optional flag, default False. Services can use this flag to decide to
  include additional information. Also, when this flag is set, the response
  contains an additional field `debug` with general information, such as
  timing of the services and the resolved ordering among tasks.
* `lang`: 2 letter language of the text, optional. Default: autodetect
* `previous`: see Resumable requests
* `tasks`: a list of requested tasks, with the shape

```json
  "tasks": [
    {"task": "parse", "name": "spacy"},
    {"task": "ner", "name": "allen"},
    {"task": "dates", "name": "misc"}
  ]
```

plus possibly other service-dependent fields.

### 1.8.1. Resumable requests

Say there are two tasks, task A and task B. Task A has a dependency on B, which
is much slower. When trying various implementations for A, it does not make
sense to recompute the result of task B again and again. In this case, one
may want to issue a request for task B, and then a second request for task A,
*passing the result* of the previous request. In this way, there will be no
need to recompute the result of task B.

In this case, one can put a field called `previous` in the request. The
content of the field must match the response for the previous request. In this
case, the server will resume computation from that point. For instance, a
user request may look like this:

```json
{
  "text": "Ulisse Dini (Pisa, 14 novembre 1845 ...",
  "tasks": [
    {"task": "names", "name": "misc"}
  ],
  "previous": {
    "ner": [
      {
        "text": "Ulisse Dini",
        "start": 0,
        "end": 11,
        "label": "PER"
      },
      ...
    ]
  }
}
```

In this example, the `ner` step is already computed, and does not need to be
recomputed again.

## 1.9. Describing services

Each service can be self describing by ovverriding the method `describe(self)`
of the `Service` class. This can be used to report information about
supported languages, dependencies, additional parameters needed in the request,
trained models and so on. The class `Service` already defines a basic
implementation, while services can add more specific information. Some
standard keys to use for this purpose are:

* `langs`: the supported languages; use `['*']` if any languages are supported
* `extra-params`: an optional list of additional parameters of the request
  accepted by the service (see example)
* `models`: a dictionary containing the information about the models used
  by the service

For each models, the following parameters are standardized:

* `pretrained`: indicates that the model is included in the library
* `trained-at`: datetime in ISO format
* `training-time`: as format `HH:mm:ss`
* `datasets`: list of datasets on which the model is trained
* `metrics`: a dictionary of metrics that measure the performance of the model
* `params`: a dictionary of parameters that were used to train the model

A complete example of response could look like this:

```python
{
  'task': 'some-task',
  'name': 'my-name',
  'deps': ['parse'],
  'optional_deps': ['ner'],
  'langs': ['it', 'en'],
  'extra-params': [
    {
      'name': 'some-param1',
      'type': 'string',
      'required': False
    },
    {
      'name': 'some-param2',
      'type': 'int',
      'required': True
    },
    {
      'name': 'some-param3',
      'type': 'string',
      'choices': ['value1', 'value2'],
      'required': True
    }
  ],
  'models': {
    'it': {
      'pretrained': False,
      'trained-at': '2019-03-27T16:00:49',
      'training-time': '02:35:23',
      'datasets': ['some-dataset'],
      'metrics': {
        'accuracy': 0.935,
        'precision': 0.87235,
        'recall': 0.77253
      },
      'params': {
        'learning-rate': 0.001,
        'momentum': 0.8,
        'num-epochs': 50
      },
    },
    'en': {
      'pretrained': True
    }
  }
}
```

You can use the `extra-params` field to describe additional parameters that
are required (or optional) for a specific service. Each extra parameter can
take the shape

```python
{
  'name': <string>,
  'type': <string>,
  'choices': <string list?>,
  'required': <bool>
}
```

where `type` can take the values `"string"` or `"int"`, and `choices` can be used
to optionally constrain the valid values for the parameter.

## 1.10. Services

The following services are defined. To read the interface: output types
are written inside `<>`. A trailing `?` denotes that the field is only present
when `debug` is `True` in the user request.

### 1.10.1. Parsing

Splits the text into sentences and the sentences into tokens. The interface
requires that the output has the shape

```python
[
  [
    {'start': <int>, 'end': <int>, 'text': <string?>},
    ...
  ]
]
```

### 1.10.2. NER

Finds people, organizations, dates, places and other entities in the text.
The interface requires that the output has the shape

```python
[
  {'start': <int>, 'end': <int>, 'text': <string?>, 'label': <string>},
  ...
]
```

### 1.10.3. Date extraction

Finds and parses dates in the text. The interface requires that the output has
the shape

```python
[
  {'start': <int>, 'end': <int>, 'text': <string?>, 'date': <string>},
  ...
]
```

where `date` is formatted as `yyyy-MM-dd`.

### 1.10.4. Codes extraction

Finds common codes in the text. The interface requires that the output has the shape

```python
[
  {'start': <int>, 'end': <int>, 'text': <string>, 'type': <string>, 'lang': <lang code>},
  ...
]
```

### 1.10.5. Fiscal codes

Extracts information from fiscal codes. The interface requires that the output has the shape

```python
[
  {'start': <int>,
   'end': <int>,
   'text': <string>,
   'type': <string>,
   'lang': <lang code>,
   'correct': <bool>, # if the fiscal code is formally correct
   'sex': <sex code>,
   'birthdate' <string>
  }
]
```

### 1.10.6. Extractive summarization

Extracts the sentences from the text that best summarize it. The interface
requires that the output has the shape

```python
[
  {'start': <int>, 'end': <int>, 'text': <string?>},
  ...
]
```

where the sentences are in order from most informative to least informative.

It can require additional (optional) parameters in the request:

* `num-extractive-sentences`: the number of sentences to extract

### 1.10.7. Keyword extraction

Extracts the most relevant keywords from the text. The interface
requires that the output has the shape

```python
[
  {'text': <string>},
  ...
]
```

where the keywords are in order from most to least relevant. Here we do not
use spans, since the important information is the keyword, which is probably
repeated many times across the text.

It can require additional (optional) parameters in the request:

* `num-keywords`: the number of keywords to extract

### 1.10.8. Sentiment detection

Detects the sentiment used in various sentences of the text. The interface
requires that the output has the shape

```python
[
  {'start': <int>, 'end': <int>, 'sentiment': <float>, 'text': <string?>},
  ...
]
```

where there is an entry for each sentence, and `sentiment` ranges from 0
(extremely negative) to 1 (extremely positive).

### 1.10.9. Names

Extract names and surnames of people mentioned in the text. It is a more refined
version of NER, which just retrieves entities of type PER.

The interface requires that the output has the shape

```python
[
  {'start': <int>, 'end': <int>, 'name': <string?>, 'surname': <string?>},
  ...
]
```

### 1.10.10. Topic modeling

Does a soft clustering of text (for instance using
[LDA](http://ethen8181.github.io/machine-learning/clustering_old/topic_model/LDA.html)
or similar techniques). This means that the text is associated to a
distribution over topics. Topics themselves are discovered as a word mixture
from the training data. The interface requires that the output has the shape

```python
{
  'distribution': <array[float]>,
  'best-topic': <int>,
  'best-score': <float>,
  'topics': <array[array[string]]?>,
}
```

where each topic is represented with the arrary of its most representative
words. The `topics` field is only present in debug mode.

It can require additional (optional) parameters in the request:

* `lda-model`: the name of a pretrained LDA model

### 1.10.11. Classification

Does a classification of the text in a pre-trained and finite set of possible
classes. This means that the text is associated to a distribution over possible
classes, of which we only output the most fitting. The interface requires that
the output has the shape

```python
{
  'category': <string>,
  'category_probability': <float>,
  'distribution': <map[string, float]?>
}
```

The `distribution` field is only present in debug mode.


## 1.11. How to create a new service

Create a new class in a file inside `src/services` which inherits from
`services.Service`. In this class, make sure to call the `Service` constructor
to register the service, like this:

```python
class SomeService(Service):
    def __init__(self, langs):
        Service.__init__(self, 'some-task', 'some-name', [], []) # first required deps, then optional deps
        ...
```
Override the method `def run(self, request, response)` which implements the
logic for your service. The return type for the service should be any
dictionary.

Also, override the method `describe(self)` to return information about the
service itself. A basic implementation of `describe` is in the `Service`
class, so a standard implementation would look like:

```python
def describe(self):
    result = super().describe()
    result['key'] = value
    # more keys
    return result
```

For the common keys, see the section on Describing services.

Be sure to check out the following things:

* The return type of `run` should be JSON serializable
* If your service defines a new task, make sure to document it in the README
* Otherwise, follow the type convention of existing services for the same task
* If your service requires some previous step (e.g. parsing), try to add it as
  a dependency and do not hardcode it inside the service
* If your service may benefit of some previous step (e.g. extra hints), you can
  add it as optional dependency; the main task will be performed whether or
  not the optional dependency is already scheduled, but if the optional dependency is
  scheduled anyway, it will be executed first.
* If your service requires an optional parameter in the request, add it
  in the schema validator in `src/server.py`
* If you cannot handle a certain language, raise `services.MissingLanguage`
* If you have a model that needs a training step, follow the conventions under
  `Organization`
* If you need an additional library, `pipenv install the-library`, then
  commit the new `Pipfile` and `Pipfile.lock`. Also remember to keep the
  requirements file up to date with `pipenv lock --requirements > requirements.txt`.
* Add tests as needed

## 1.12. Testing

Tests are written with [nose](https://nose.readthedocs.io/en/latest/). If you
have installed Charade in development mode (`pipenv install --dev`), you can run
tests with the `nosetests` command.

Tests for a particular service should put under `tests/services/test_the_service.py`.
The naming convention is so that Nose autodiscovery will find them when
running `nosetests`. Classes and methods should also follow this naming
convention:

```python
class TestTheThing(TestCase):
    def test_something(self):
        ...
```

You can also test here classes and functions under `common`. If you need to
test something which is only used in training, put it under `common` as well.

Tests for Charade itself are placed under `tests` without further nesting.

## 1.13. Style guide

* Follow PEP-8
* Prefer long names such as `request`, `result`, `token` over `req`, `res`, `tok`
* But be consistent with libraries: for instance, `spacy` defines `document.ents`
  Iterate over that as `for ent in documents.ents:`
* Do not use trailing commas
* Do not commit models or data - commit scripts to retrieve them
* All bash scripts use `set -e`, `set -u`
* Make sure that bash scripts can be called from anywhere (see the existing one
  for examples)

## 1.14. Organization

Follow a tree similar to the following

```
.
├── Pipfile
├── Pipfile.lock
├── README.md
├── TODO.md
├── data
│   └── ner
│       └── ...
├── examples
│   ├── request.json
│   ├── request.sh
│   ├── request2.json
│   └── request3.json
├── models
│   └── pytorch
│       └── ner
│           └── ...
├── requirements.txt
├── resources
│   ├── names
│   │   └── it.txt
│   ├── stopwords
│   │   └── en.txt
│   └── surnames
│       └── it.txt
├── scripts
│   └── pytorch
│       └── ner
│           └── it
│               ├── 1-get-data.sh
│               ├── 2-prepare-data.sh
│               └── 3-train.sh
├── src
│   ├── __init__.py
│   ├── common
│   │   ├── __init__.py
│   │   └── pytorch
│   │       ├── __init__.py
│   │       └── ner
│   │           ├── __init__.py
│   │           └── model.py
│   ├── main.py
│   ├── server.py
│   ├── services
│   │   ├── __init__.py
│   │   ├── allen.py
│   │   ├── misc.py
│   │   ├── pytorch.py
│   │   ├── regex.py
│   │   ├── spacy.py
│   │   └── textrank.py
│   └── training
│       └── pytorch
│           └── ner
│               ├── generate_wikiner_vectors.py
│               └── train.py
└── tests
    ├── __init__.py
    ├── services
    │   ├── __init__.py
    │   └── test_textrank.py
    └── test_server.py
```

It should be clear what goes where: `data`, `models`, `resources`, `training`
and so on. When in doubt, follow existing conventions. The directory `common`
holds code that should be shared at inference and training time.

Under `data`, only put data that is needed at training time - everything that
is needed at inference time goes under `models`. If some data file is needed
also at inference time, either

* store the content of the file as a field inside the model, or
* make sure that the training scripts copy the necessary files from `data` to `models`.