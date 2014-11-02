#pragma once

Steinberg::FUnknown *createVTrackEffect(void *context);

// For both track and trig, 255 means "all" (if applicable)
struct ParamId {
	uint8_t type;
	union {
		uint8_t trig;
		uint8_t direct_out;
	};
	union {
		uint8_t track;
		uint8_t direct_in;
	};

	ParamId(): type(), trig(), track() {}
	ParamId(uint32_t id): type(id), trig(id >> 8), track(id >> 16) {}

	bool trig_related() const {
		return !!(type & 0x80);
	}

	bool midi_trig_related() const {
		return (type & 0xc0) == 0xc0; // type >> 6 == 3?
	}

	uint32_t encode() {
		return (track << 16) | (trig << 8) | type;
	}
	static ParamId decode(uint32_t id) {
		return ParamId(id);
	}
};

enum ParamIdType {
	kParamArm,
	// direct_in, direct_out identify the input and output channels in question
	kParamDirectLevel,
	kParamLevel,

	// Trig parameters - 128 and up, MIDI trig parameters - 192 and up.
	kParamMidiTrigEnable = 192,
	kParamMidiTrigNote,
	kParamMidiTrigLength,
	kParamSampleTrigEnable = 128,
	//kParamSampleTrigOneShot,
	kParamSampleTrigStack,
	kParamSampleTrigSampleNumber,
	kParamLatchTrigEnable,
	kParamLatchTrigOneShot,
};