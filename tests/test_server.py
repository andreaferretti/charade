from unittest import TestCase
from nose.tools import raises

from server import _resolve
from services import Service, MissingService


def mk_services(services):
    result = {}
    for service in services:
        task = service.task
        name = service.name
        if not task in result:
            result[task] = {}
        result[task][name] = service
    return result

class TestTaskResolution(TestCase):
    services = mk_services([
        Service('task1', 'name1', []),
        Service('task2', 'name2', []),
        Service('task3', 'name1', ['task1']),
        Service('task1', 'name2', []),
        Service('task4', 'name2', ['task3', 'task2']),
        Service('task5', 'name1', [], ['task1'])
    ])

    # Simple example of task resolution
    def test_simple(self):
        request = {
            'tasks': [
                {'task': 'task3', 'name': 'name1'}
            ]
        }
        response = {}
        resolution = _resolve(request, response, TestTaskResolution.services)
        expected = [('task1', 'name1'), ('task3', 'name1')]
        self.assertEqual(resolution, expected)

    # Recursive task resolution
    def test_recursive(self):
        request = {
            'tasks': [
                {'task': 'task4', 'name': 'name2'}
            ]
        }
        response = {}
        resolution = _resolve(request, response, TestTaskResolution.services)
        expected = [('task1', 'name1'), ('task3', 'name1'), ('task2', 'name2'), ('task4', 'name2')]
        self.assertEqual(resolution, expected)

    # Task resolution under the constraint that a specific
    # version of task1 was requested by the user
    def test_explicit(self):
        request = {
            'tasks': [
                {'task': 'task3', 'name': 'name1'},
                {'task': 'task1', 'name': 'name2'}
            ]
        }
        response = {}
        resolution = _resolve(request, response, TestTaskResolution.services)
        expected = [('task1', 'name2'), ('task3', 'name1')]
        self.assertEqual(resolution, expected)

    # Task resolution assuming the response is partially
    # provided by the user
    def test_partial_response(self):
        request = {
            'tasks': [
                {'task': 'task3', 'name': 'name1'},
                {'task': 'task2', 'name': 'name2'}
            ]
        }
        response = {
            'task1': []
        }
        resolution = _resolve(request, response, TestTaskResolution.services)
        expected = [('task1', None), ('task3', 'name1'), ('task2', 'name2')]
        self.assertEqual(resolution, expected)

    # Task resolution when one of the tasks has an optional dependence.
    # if the optional dependence is requested, it should be performed in advance
    def test_optional_dependence(self):
        request0 = {
            'tasks': [
                {'task': 'task5', 'name': 'name1'},
                {'task': 'task1', 'name': 'name2'}
            ]
        }
        request1 = {
            'tasks': [
                {'task': 'task5', 'name': 'name1'}
            ]
        }
        response = {}
        resolution0 = _resolve(request0, response, TestTaskResolution.services)
        resolution1 = _resolve(request1, response, TestTaskResolution.services)
        expected0 = [('task1','name2'), ('task5','name1')]
        expected1 = [('task5','name1')]
        self.assertEqual((resolution0, resolution1), (expected0, expected1))

    

    # When services are missing, appropriate exceptions are raised
    @raises(MissingService)
    def test_missing_service(self):
        request = {
            'tasks': [
                {'task': 'task3', 'name': 'name1'},
                {'task': 'task2', 'name': 'name2'},
                {'task': 'taskx', 'name': 'namex'}
            ]
        }
        response = {}
        resolution = _resolve(request, response, TestTaskResolution.services)