"""Group references without turning extra views into extra people (no ML imports)."""

MAX_PHOTOS = 8


def identity_consensus(embeddings):
    """One identity vector, with primary appearance handled separately.

    Average unit directions so embedding magnitude cannot make one photo
    dominate, then restore the mean input norm expected by the trained adapter.
    This is experimental pooling, not a trained multi-view identity encoder.
    """
    import math
    vectors = [list(map(float, row)) for row in embeddings]
    if not vectors or not vectors[0] or any(len(v) != len(vectors[0]) for v in vectors):
        raise ValueError("Identity embeddings must have equal, nonzero dimensions.")
    norms = [math.sqrt(sum(x*x for x in v)) for v in vectors]
    if any(not math.isfinite(n) or n <= 1e-8 for n in norms):
        raise ValueError("Identity embeddings must be finite and nonzero.")
    if len(vectors) == 1:
        return vectors[0]
    direction = [sum(v[i] / n for v, n in zip(vectors, norms)) / len(vectors)
                 for i in range(len(vectors[0]))]
    norm = math.sqrt(sum(x*x for x in direction))
    if norm <= 1e-8:
        raise ValueError("Reference identity features disagree; choose a consistent photo set.")
    scale = sum(norms) / len(norms) / norm
    return [x * scale for x in direction]


def grouped_references(faces, extras, regions):
    """Flatten photos with repeated regions for upstream's reference attention mask.

    Each entry is (image, box, person_number, photo_number). The model gets
    separate reference features, all attending to the same person's region.
    No embedding averaging, resizing to a common batch, or extra face positions.
    """
    present = [i for i, face in enumerate(faces) if face is not None]
    if present != list(range(len(present))) or len(regions) != len(present):
        raise ValueError("WithAnyone needs one face position per person, with no empty person slots.")
    if not present or len(present) > 4:
        raise ValueError("WithAnyone needs one to four people.")
    if any(extra and faces[i] is None for i, extra in enumerate(extras)):
        raise ValueError("Extra reference photos need a person's primary photo.")
    result = []
    for i in present:
        photos = [faces[i], *(extras[i] or ())]
        if len(photos) > MAX_PHOTOS:
            raise ValueError("Person %d has more than %d reference photos." % (i + 1, MAX_PHOTOS))
        for j, photo in enumerate(photos, 1):
            if len(photo.shape) != 4 or photo.shape[0] != 1:
                raise ValueError("Person %d reference %d must be one still image." % (i + 1, j))
            result.append((photo, regions[i], i + 1, j))
    return result
