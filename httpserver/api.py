from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib import parse
import json

from httpserver import utils
from httpserver import controller

PORT = 8080

class TrackingHTTPRequestHandler(BaseHTTPRequestHandler):

    # Predefinitions about the expected parameters
    modeparam = 'mode'
    actions = {
        'starttracking': controller.start,
        'stoptracking': controller.stop,
        'clear': controller.clear,
        'gettracking': controller.get,
        'getlist': controller.getlist
    }
    paramvalidation = {
        'starttracking': {
            'furl': {
                'type': 'str',
            },
            'shape': {
                'type': 'int',
                'size': 2
            },
            'bbox': {
                'type': 'int',
                'size': 4
            },
            'timestamp': {
                'type': 'int'
            }
        },
        'stoptracking': {'id': {'type': 'int'}},
        'clear': {},
        'gettracking': {'id': {'type': 'int'}},
        'getlist': {}
    }
    # Predefined error responses
    responses = {
        'action_not_found': {
            'success': False,
            'message': 'Mode empty or undefined.'
        },
        'params_validation_error': {
            'success': False,
            'message': 'Invalid parameters.'
        }
    }

    def respond(self, response, jsonresponse=True, status=200):
        self.send_response(status)
        self.end_headers()
        if response is not None:
            if jsonresponse:
                body = json.dumps(response)
            else:
                body = response
            self.wfile.write(body.encode())

    def do_GET(self):
        url = parse.urlparse(self.path)
        if url.path == '/api':
            querydict = parse.parse_qs(url.query)
            if self.modeparam in querydict:
                act = querydict[self.modeparam][0]
                if act in self.actions:
                    if utils.validate(querydict, self.paramvalidation[act]):
                        valid,parsedquery = utils.parseparams(
                            querydict,
                            self.paramvalidation[act])
                        if valid:
                            response = self.actions[act](parsedquery)
                        else:
                            response = self.responses['params_validation_error']
                    else:
                        response = self.responses['params_validation_error']
                else:
                    response = self.responses['action_not_found']
            else:
                response = self.responses['action_not_found']
            self.respond(response)
        elif url.path == '/client.sample.html':
            debugfile = open('httpserver' + url.path, 'r').read()
            self.respond(debugfile, False)
        else:
            self.respond('Not Found.', False, 404)

if __name__ == '__main__':
    controller.init()

    server = HTTPServer(('', PORT), TrackingHTTPRequestHandler)
    server.serve_forever()
