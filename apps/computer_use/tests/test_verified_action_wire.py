"""Host schema qualification against malformed shapes from actual model runs."""
import unittest

from jsonschema import Draft202012Validator
from qw3_computer.tool_schema import tools_for_platform


class VerifiedActionWireTests(unittest.TestCase):
    def request(self, action, **extra):
        return {'desktop_id':'d','observation_id':'o','action_id':'wire1','action':action,**extra}

    def test_verified_platforms_reject_wrong_or_missing_coordinate_fields(self):
        for platform in ('windows','linux','android'):
            schema=tools_for_platform(platform,workflow=True)[1]['inputSchema']
            Draft202012Validator.check_schema(schema)
            validate=Draft202012Validator(schema).is_valid
            for action in ({'kind':'key','key':'ENTER'},{'kind':'type','text':'literal'},
                           {'kind':'open_app','app':'settings' if platform=='android' else 'calc'}):
                self.assertTrue(validate(self.request(action)))
                self.assertFalse(validate(self.request(action,coordinate_space='normalized_1000')))
            click={'kind':'click','x':200,'y':200,'clicks':1,'button':'left'}
            self.assertFalse(validate(self.request(click)))
            self.assertFalse(validate(self.request(click,coordinate_space='pixels')))
            self.assertTrue(validate(self.request(click,coordinate_space='normalized_1000')))
            self.assertFalse(validate(self.request(click|{'x':1001},coordinate_space='normalized_1000')))
            self.assertEqual(schema['properties']['coordinate_space']['type'],'string')

    def test_verified_description_does_not_advertise_legacy_pixel_behavior(self):
        for platform in ('windows','linux','android'):
            description=tools_for_platform(platform,workflow=True)[1]['description']
            self.assertNotIn('Omitted coordinate_space retains pixel behavior',description)
            self.assertNotIn('or pixels for accessibility control centers',description)
        legacy=tools_for_platform('linux')[1]
        self.assertIn('Omitted coordinate_space retains pixel behavior',legacy['description'])
        self.assertNotIn('oneOf',legacy['inputSchema'])


if __name__=='__main__':unittest.main()
