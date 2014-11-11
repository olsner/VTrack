#include "stdafx.h"
#include "VTrackController.h"
#include "VTrackEffect.h"

using namespace Steinberg;
using namespace Steinberg::Vst;

class VTrackController: public EditControllerEx1 /*, public IMidiMapping*/ {
public:
	VTrackController() {
		HERE;
	}

	~VTrackController() {
	}

	IPlugView* PLUGIN_API createView(FIDString name) override {
		if (strcmp(name, "editor") == 0) {
			return new VSTGUI::VST3Editor(this, "view", "vtrack.uidesc");
		}
		return 0;
	}
};

Steinberg::FUnknown *createVTrackController(void *context) {
	return (IEditController*)new VTrackController();
}
