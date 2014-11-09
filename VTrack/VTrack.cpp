// VTrack.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "VTrack.h"
#include "VTrackEffect.h"

#include "againcontroller.h" // for AGainController
#include "againcids.h"

#define stringPluginName "VTrack"
#define FULL_VERSION_STR "0.1"

bool InitModule()
{
	return true;
}

bool DeinitModule()
{
	return true;
}

using namespace Steinberg::Vst;

#define MY_DEF_CLASS(fuid, kind, name, constructor) \
		DEF_CLASS2(INLINE_UID_FROM_FUID(fuid), PClassInfo::kManyInstances, kind, name, Vst::kDistributable, "Instrument|Sampler", FULL_VERSION_STR, kVstVersionString, constructor)

BEGIN_FACTORY_DEF("VTrack", "http://vtrack.olsner.se", "mailto:olsner@gmail.com") {

	HERE;
	assert(VTrackProcessorUID.isValid());
	VTrackProcessorUID.print(0, FUID::UIDPrintStyle::kFUID);
	assert(VTrackControllerUID.isValid());
	VTrackControllerUID.print(0, FUID::UIDPrintStyle::kFUID);
	/*FUID VTrackProcessorUID;
	VTrackProcessorUID.generate();
	VTrackProcessorUID.print(0, FUID::UIDPrintStyle::kFUID);
	FUID VTrackControllerUID;
	VTrackControllerUID.generate();
	VTrackControllerUID.print(0, FUID::UIDPrintStyle::kFUID);*/

	MY_DEF_CLASS(AGainProcessorUID, kVstAudioEffectClass, "VTrack", createVTrackEffect);
	MY_DEF_CLASS(VTrackControllerUID, kVstComponentControllerClass, "VTrackController", Steinberg::Vst::AGainController::createInstance/*TODO*/);
}
END_FACTORY
