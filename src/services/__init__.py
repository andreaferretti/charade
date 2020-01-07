class Service:
    def __init__(self, task, name, deps=[], optional_deps=[]):
        self.task = task
        self.name = name
        self.deps = deps
        self.optional_deps = optional_deps

    def run(self, request, response):
        pass

    def describe(self):
        return {
            'task': self.task,
            'name': self.name,
            'deps': self.deps,
            'optional_deps': self.optional_deps
        }

class ApiError(Exception):
    def __init__(self, message):
        Exception.__init__(self)
        self.message = message

class MissingLanguage(ApiError):
    def __init__(self, lang):
        ApiError.__init__(self, f'Missing language `{lang}`')

class MissingService(ApiError):
    def __init__(self, task, name):
        ApiError.__init__(self, f'Missing task `{task}` with implementation `{name}`')

class MissingModel(ApiError):
    def __init__(self, task, name, model, models=None):
        ApiError.__init__(self, f'Missing model `{model}` for task `{task}` with implementation `{name}`' +
                          ('. Available models are: ' + ', '.join('`'+m+'`' for m in models) if models else ''))

class MissingParameter(ApiError):
    def __init__(self, task, name, param):
        ApiError.__init__(self, f'Missing required parameter `{param}` for task `{task}` with implementation `{name}`')

class MissingResource(ApiError):
    def __init__(self, task, name, resource):
        ApiError.__init__(self, f'Missing resource `{resource}` for task `{task}` with implementation `{name}`')