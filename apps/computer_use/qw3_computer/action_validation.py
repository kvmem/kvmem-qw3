"""Point to the exact malformed field without changing or dispatching input."""
import re
from .tool_schema import TOOLS


class ActionValidationError(ValueError):
    def __init__(self, code, path, message, hint):
        self.problem = {'code': code, 'path': path, 'message': message, 'hint': hint}
        super().__init__(f'{path}: {message} {hint}')


def fail(code, path, message, hint='Correct the field and submit a new tool call; no input was dispatched.'):
    raise ActionValidationError(code, path, message, hint)


def validate_action(arguments):
    """Validate the public act schema before acquiring a delivery identity.

    This deliberately rejects misplaced fields instead of interpreting the
    model's intent. It implements the small schema vocabulary used by act.
    """
    def check(value, schema, path):
        if 'oneOf' in schema:
            if not isinstance(value, dict):
                fail('wrong_type', path, 'Expected an object.')
            branches = schema['oneOf']
            selected = next((s for s in branches if s['properties']['kind']['const'] == value.get('kind')), None)
            if selected is None:
                allowed = ', '.join(s['properties']['kind']['const'] for s in branches)
                fail('invalid_action_kind', path + '.kind', 'Unknown or missing action kind.', 'Use one of: ' + allowed)
            return check(value, selected, path)
        expected = schema.get('type')
        if expected == 'object':
            if not isinstance(value, dict):
                fail('wrong_type', path, 'Expected an object.')
            for name in schema.get('required', []):
                if name not in value:
                    hint = 'Required here, alongside ' + ', '.join(schema.get('properties', {})) + '.'
                    if path == '$' and isinstance(value.get('action'), dict) and name in value['action']:
                        hint = f'Move $.action.{name} to $.{name}; it must be a sibling of action, not inside action.'
                    fail('missing_field', path + '.' + name, 'Required field is missing.', hint)
            for name, member in value.items():
                if name not in schema.get('properties', {}):
                    fail('unexpected_field', path + '.' + name, 'This field is not accepted at this location.')
                check(member, schema['properties'][name], path + '.' + name)
        elif expected == 'string':
            if not isinstance(value, str):
                fail('wrong_type', path, 'Expected a string.')
            if len(value) < schema.get('minLength', 0) or len(value) > schema.get('maxLength', float('inf')):
                fail('invalid_length', path, 'String length is outside the accepted range.',
                     f"Use {schema.get('minLength', 0)}..{schema.get('maxLength', 'unbounded')} characters.")
            if 'pattern' in schema and not re.fullmatch(schema['pattern'], value):
                fail('invalid_characters', path, 'String contains unsupported characters.',
                     'Use letters, digits, underscores or hyphens for this identifier.')
        elif expected == 'integer':
            if type(value) is not int:
                fail('wrong_type', path, 'Expected an integer, not a quoted string, boolean or decimal.')
            if value < schema.get('minimum', -float('inf')) or value > schema.get('maximum', float('inf')):
                fail('out_of_range', path, 'Number is outside the accepted range.',
                     f"Use {schema.get('minimum', '-unbounded')}..{schema.get('maximum', 'unbounded')}.")
        if 'const' in schema and value != schema['const']:
            fail('invalid_value', path, 'Unexpected constant value.')
        if 'enum' in schema and value not in schema['enum']:
            fail('invalid_value', path, 'Unsupported value.', 'Use one of: ' + ', '.join(map(str, schema['enum'])))
    check(arguments, TOOLS[1]['inputSchema'], '$')
