# HTML cleaning

HTML cleaning (extracting the text of the actual content) is an interesting
task. We have a Scala implementation that should be ported to Python

# Wikification

As in [TAGME](http://pages.di.unipi.it/ferragina/cikm2010.pdf)

# Extract logic related to BIO tags and similar schemes and test it

At the moment, there is come custom logic to translate BIO tags inside
`services/allen.py`, we should generalize this and support other schemes,
test this and move it in `services/__init__.py` for consumption by other
models.