import unittest

from apps.image_studio.scene import smile
from apps.image_studio import imagegen as ig


class TestSmile(unittest.TestCase):
    def test_expression_shape_and_negative_intent(self):
        for text, want in [('soft smile', 'closed'), ('closed-mouth smile', 'closed'),
                           ('broad smile', 'broad'), ('grinning', 'broad'),
                           ('laughing', 'laugh'), ('smiling', 'smile'),
                           ('neutral', ''), ('not smiling', ''), ('unsmiling', ''),
                           ('serious', ''), ('without a smile', '')]:
            self.assertEqual(smile.intent(text), want, text)
        self.assertEqual(smile.requests({'scene': 'A smiling fox in snow'}), [])
        self.assertEqual(smile.requests({'scene': 'A smiling man', 'expression': 'neutral'}), [])

    def test_scene_subjects_keep_their_own_expressions(self):
        s = {'scene': 'A smiling woman beside a serious man', 'scene_faces': {'people': [
            {'id': 'a', 'region': [0, 0, .4, .5], 'expression': 'broad smile'},
            {'id': 'b', 'region': [.6, 0, 1, .5], 'expression': 'serious'}]}}
        wanted = smile.requests(s)
        self.assertEqual(len(wanted), 1)
        self.assertEqual(wanted[0]['id'], 'a')
        # Older face targets recover expression from the saved scene object.
        s['scene_faces']['people'][0].pop('expression')
        s['scene_layout'] = {'objects': [{'id': 'a', 'look': {'expression': 'soft smile'}}]}
        self.assertEqual(smile.requests(s)[0]['shape'], 'closed')

    def test_scene_generation_carries_each_persons_expression(self):
        from apps.image_studio.scene import scene as sc
        scene = sc.new_scene()
        person = sc.new_object('person')
        person['look']['expression'] = 'soft smile'
        scene['objects'].append(person)
        targets = sc.face_targets(scene)
        self.assertEqual(targets[0]['expression'], 'soft smile')
        self.assertEqual(smile.requests({'scene_faces': {'people': targets}})[0]['shape'], 'closed')

    def test_region_and_point_select_only_the_intended_mouth(self):
        faces = [(100, 100, 100, 100), (700, 100, 100, 100)]
        wanted = [{'shape': 'closed', 'region': [.6, 0, 1, .5]}]
        spots, skipped = smile.spots(1000, 1000, wanted, faces)
        self.assertEqual(skipped, 0)
        self.assertEqual(spots[0]['face'], faces[1])
        self.assertEqual(spots[0]['box'], [712, 158, 76, 33])
        wanted[0]['region'] = None
        spots, skipped = smile.spots(1000, 1000, wanted, faces,
                                     [{'target_point': [.15, .15]}])
        self.assertEqual(spots[0]['face'], faces[0])
        self.assertEqual(skipped, 0)

    def test_ambiguous_missing_or_off_region_faces_are_not_guessed(self):
        faces = [(100, 100, 100, 100), (700, 100, 100, 100)]
        for region, boxes in [(None, faces), ([0, .8, 1, 1], faces), (None, [])]:
            self.assertEqual(smile.spots(1000, 1000,
                [{'shape': 'smile', 'region': region}], boxes), ([], 1))

    def test_smile_crop_redraws_only_the_mouth_band_and_preserves_loras(self):
        spots, _ = smile.spots(512, 512, [{'shape': 'smile'}], [(200, 100, 80, 100)])
        crops = smile.crops(512, 512, spots)
        wf = ig.load_workflow('klein9b_base')
        g = ig.face_graph(wf, dict(wf['defaults'], model='m', encoder='e', vae='v',
            prompt='A smiling man', face_prompt=spots[0]['prompt'], seed=1,
            face_denoise=smile.DENOISE, redraw_steps=20), [('identity.safetensors', .8)],
            'made.png', crops, 'oval.png', 'out')
        self.assertEqual(g['fc1_4']['inputs']['latent_image'], ['fc1_3n', 0])
        self.assertEqual(g['fc1_3n']['inputs']['mask'], ['fc1_a2', 0])
        self.assertEqual(g['fc1_guider']['inputs']['model'], ['lora1', 0])
        self.assertNotIn('fc1_s0', g)

    def test_smile_crop_uses_integer_pixel_coordinates(self):
        # Odd mouth bands have half-pixel centers; ComfyUI slices cannot use floats.
        wf = ig.load_workflow('klein9b_base')
        for face in [(200, 100, 80, 100), (201, 101, 81, 101)]:
            spots, _ = smile.spots(512, 512, [{'shape': 'smile'}], [face])
            crops = smile.crops(512, 512, spots)
            g = ig.face_graph(wf, dict(wf['defaults'], model='m', encoder='e', vae='v',
                prompt='A smiling man', face_prompt=spots[0]['prompt'], seed=1,
                face_denoise=smile.DENOISE), [], 'made.png', crops, 'oval.png', 'out')
            region = g['fc1_1']['inputs']['crop_region']
            self.assertTrue(all(type(value) is int for value in region.values()), region)


if __name__ == '__main__':
    unittest.main()
