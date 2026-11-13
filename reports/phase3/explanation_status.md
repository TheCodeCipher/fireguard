# Explanation layer status (Role 4, P3-R4-06)

**UNAVAILABLE.** No feature-group ablation, permutation, gradient or Grad-CAM explanation has been
computed, and none has been tested for faithfulness or stability. The replay app must not show an
explanation layer. If one is added later, it must first pass a faithfulness test (deleting the
top-attributed region changes the prediction more than deleting a random region of equal size) and
a stability test across seeds and two methods on a fixed sample set, and it must be described as
an attribution, never as a cause.
