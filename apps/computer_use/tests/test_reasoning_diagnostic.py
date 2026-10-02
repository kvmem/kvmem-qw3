"""Protect the P2 comparison from leaked answers and permissive grading."""
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from reasoning_diagnostic_run import grade,request_for,strict_equal
from reasoning_diagnostic_materials import CASES


class ReasoningDiagnosticTests(unittest.TestCase):
    def test_only_modality_and_actual_thinking_change_between_matched_requests(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);Image.new('RGB',(1280,800),'white').save(root/'card.png')
            case={'image':'card.png','question':'Choose one.','schema':{'choice':'string'},'equivalent_text':'Visible rules only.',
                  'expected':{'choice':'SECRET_HOST_ONLY_ANSWER'}}
            bodies=[request_for(case,root,p,t,4096) for p in ('screenshot','text') for t in (False,True)]
            for body in bodies:
                self.assertNotIn('SECRET_HOST_ONLY_ANSWER',json.dumps(body))
                self.assertEqual(body['max_tokens'],4096)
            self.assertEqual(bodies[0]|{'enable_thinking':True},bodies[1])
            self.assertEqual(bodies[2]|{'enable_thinking':True},bodies[3])
            self.assertEqual(bodies[0]['messages'][0]['content'][0],bodies[2]['messages'][0]['content'][0])

    def test_failed_constraints_and_invalid_types_do_not_pass(self):
        case={'expected':{'can_submit':False,'required':['manager','security']}}
        self.assertTrue(grade(case,'{"required":["manager","security"],"can_submit":false}')['passed'])
        for answer in ('{"required":["manager"],"can_submit":false}',
                       '{"required":["manager","security"],"can_submit":0}',
                       '{"required":["manager","security"],"can_submit":false,"extra":true}'):
            self.assertFalse(grade(case,answer)['passed'])
        self.assertFalse(strict_equal({'amounts':{'A':'1.00'}},{'amounts':{'A':1}}))

    def test_purchase_answer_has_unique_independent_enumerated_optimum(self):
        candidates=[]
        for a in range(3):
            for b in range(5):
                for c in range(19):
                    subtotal=215*a+152*b+43*c;payable=subtotal+(35 if subtotal<700 else 0)
                    if 6*a+4*b+c==18 and payable<=710:candidates.append((payable,a+b+c,-a,a,b,c))
        best=min(candidates)
        expected=next(c['expected'] for c in CASES if c['id']=='calculation-2')
        self.assertEqual(best[3:],(expected['A'],expected['B'],expected['C']))
        self.assertEqual(best[0],int(float(expected['payable'])))
        self.assertEqual(sum(c[0]==best[0] for c in candidates),1)
