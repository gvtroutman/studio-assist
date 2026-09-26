"""Profile selection through to the saved result, without models or network."""
import os
import unittest
from unittest.mock import patch

import studio_facefusion as ff
import studio_imagegen as ig
from test_imagegen import TempStudioMixin, PNG, settle


class TestProfiles(TempStudioMixin, unittest.TestCase):
    def profile(self):
        path = os.path.join(self.dir, 'reference.png')
        with open(path, 'wb') as file:
            file.write(PNG)
        self.studio.lib.save('identities', [{'id': 'lilya', 'name': 'Lilya',
            'references': [path], 'use_references': False, 'avatar': 'generated-avatar.png'}])
        return self.studio.lib.get('identities', 'lilya')

    def test_selected_profile_applies_after_refinement_and_is_recorded(self):
        profile = self.profile()
        order = []
        def refine(job, client, plan, values, files, say):
            order.append('refine')
            return files
        def swap(data, who, **kw):
            order.append('swap')
            self.assertEqual(who['references'], profile['references'])
            self.assertNotIn(who['avatar'], who['references'])
            return PNG, {'outside_mask_changed_pixels': 0, 'identity': who['id']}
        with patch.object(self.studio, '_refine', refine), patch.object(ff, 'available', return_value=True), \
                patch.object(ff, 'swap', side_effect=swap):
            jobs = self.studio.submit(dict(ig.default_settings(), scene='Dancing',
                model='z-image-turbo', backend='5090', identities=[{'id': 'lilya'}], auto_refine=True))
            settle(jobs)
        self.assertEqual(jobs[0].status, 'complete', jobs[0].detail)
        self.assertEqual(order, ['refine', 'swap'])
        self.assertEqual(jobs[0].record['facefusion'][0]['identity'], 'lilya')

    def test_missing_facefusion_fails_before_rendering(self):
        self.profile()
        with patch.object(ff, 'available', return_value=False):
            jobs = self.studio.submit(dict(ig.default_settings(), scene='Dancing',
                model='z-image-turbo', backend='5090', identities=['lilya']))
            settle(jobs)
        self.assertEqual(jobs[0].status, 'failed')
        self.assertIsNone(jobs[0].graph)
        self.assertIn('FaceFusion', jobs[0].detail)

    def test_no_selection_does_not_apply_a_profile(self):
        self.profile()
        self.assertEqual(ff.selected(self.studio.lib, {'identities': []}), [])
        self.assertEqual(len(ff.selected(self.studio.lib, {'identities': ['lilya', 'lilya']})), 1)


if __name__ == '__main__':
    unittest.main()
