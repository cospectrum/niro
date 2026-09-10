"""Keep generated checks bounded without timing-dependent failures on slow CI."""

import hypothesis

hypothesis.settings.register_profile("property", max_examples=200, deadline=None)
hypothesis.settings.load_profile("property")
