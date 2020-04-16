import time

from langdetect import detect
from bottle import Bottle, request, static_file, hook, response
from cerberus import Validator, DocumentError
from networkx import Graph, topological_sort

from services import ApiError, MissingService

_schema = {
    'text': {'type': 'string', 'required': True},
    'tasks': {
        'type': 'list',
        'required': True,
        'schema': {
            'type': 'dict',
            'schema': {
                'task': {'type': 'string', 'required': True},
                'name': {'type': 'string', 'required': True}
            }
        }
    },
    'lang': {'type': 'string', 'minlength': 2, 'maxlength': 2},
    'debug': {'type': 'boolean'},
    'previous': {
        'type': 'dict',
        'allow_unknown': True
    },
    # Task-dependent fields
    'num-keywords': {'type': 'integer'},
    'num-extractive-sentences': {'type': 'integer'},
    'lda-model': {'type': 'string'},
    'bert-model': {'type': 'string'},
    'nmf-model': {'type': 'string'},
    'classification-model': {'type': 'string'},
    'target-lang': {'type': 'string', 'minlength': 2, 'maxlength': 2}
}

# Resolve the graph of service dependencies


def _resolve(request, response, services):
    # Create the graph of tasks and sort it topologically
    g = Graph().to_directed()
    # Mapping task -> chosen implementation
    tasks = dict()
    # Tasks for which we still have to consider the dependencies
    todo = set()
    for item in request['tasks']:
        task = item['task']
        name = item['name']
        g.add_node(task)
        tasks[task] = name
        todo.add(task)
    while len(todo) > 0:
        todo_next = set()
        for task in todo:
            name = tasks[task]
            service = services.get(task, {}).get(name)
            if service is None:
                raise MissingService(task, name)
            deps = service.deps
            for dep in deps:
                if not dep in tasks:
                    g.add_node(dep)
                    if dep in response:
                        tasks[dep] = None
                    else:
                        if not dep in services:
                            # Should not happen, deps should be always available
                            raise MissingService(dep, 'any')
                        dep_service = next(s for s in services[dep].values())
                        dep_name = dep_service.name
                        tasks[dep] = dep_name
                        todo_next.add(dep)
                g.add_edge(dep, task)
            optional_deps = service.optional_deps
            for optional_dep in optional_deps:
                if optional_dep in tasks:
                    g.add_edge(optional_dep, task)
        todo = todo_next

    return [(task, tasks[task]) for task in topological_sort(g)]


class Server:
    def __init__(self):
        self.services = {}
        self.app = Bottle()
        # TODO: check if validator is thread-safe, otherwise
        # initialize it once per-request.
        self.validator = Validator()
        self.validator.schema = _schema

    def add(self, service):
        task = service.task
        name = service.name
        if not task in self.services:
            self.services[task] = {
                name: service
            }
        else:
            self.services[task][name] = service

    def handle(self, request):
        response = request.get('previous', {})
        if not 'lang' in request:
            lang = detect(request['text'])
            request['lang'] = lang
            response['lang'] = lang
        last_time = time.time()
        times = {}

        ordered_tasks = _resolve(request, response, self.services)
        for task, name in ordered_tasks:
            # None happens when the field is already in the response
            if name is not None:
                service = self.services[task][name]
                response[task] = service.run(request, response)
                now = time.time()
                times[task] = now - last_time
                last_time = now

        if request.get('debug', False):
            response['debug'] = {
                'timing': times,
                'task-ordering': ordered_tasks
            }

        # Some intermediate tasks may have not been explicitly asked
        # by the client
        keys_to_keep = [item['task'] for item in request['tasks']] + ['debug']
        return {k: v for k, v in response.items() if k in keys_to_keep}

    def run_http(self, host='localhost', port=9000, debug=False, wsgi=False):

        if debug:
            @self.app.hook('after_request')
            def enable_cors():
                response.headers['Access-Control-Allow-Methods'] = '*'
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type'

            @self.app.route('/', method='OPTIONS')
            def cors():
                return ''

        @self.app.post('/')
        def run_inner():
            req = request.json
            try:
                valid = self.validator.validate(req)
                if not valid:
                    return {
                        'errors': self.validator.errors
                    }
            except DocumentError:
                return {
                    'error': 'Malformed request'
                }
            try:
                return self.handle(req)
            except ApiError as e:
                return {
                    'error': e.message
                }

        @self.app.get('/')
        def document():
            response = {}
            for task, d in self.services.items():
                response[task] = []
                for name, service in d.items():
                    response[task].append(service.describe())

            return {'services': response}

        @self.app.get('/app')
        def site():
            return static_file('index.html', root='app')

        @self.app.get('/static/<filepath:path>')
        def server_static(filepath):
            return static_file(filepath, root='app/static/')

        if not wsgi:
            self.app.run(host=host, port=port, debug=debug, reloader=debug)

        return self.app
