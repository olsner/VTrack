#include "stdafx.h"

#include "VTrackEffect.h"
#include "againcids.h"
#include "againparamids.h"

using namespace Steinberg::Vst;
using namespace Steinberg;
using std::vector;
using std::unique_ptr;
using std::cerr;

#define SAMPLE_RATE 44100
#define NUM_INPUTS 4
#define NUM_OUTPUTS 4
#define NUM_TRACKS 8
#define PATTERN_SCALE 16 // 16 = 16th notes
#define TRIGS_PER_QN (PATTERN_SCALE / 4)
#define PATTERN_LENGTH_QN 4
#define PATTERN_LENGTH (PATTERN_LENGTH_QN * TRIGS_PER_QN)
#define MAX_STACK_SIZE 16 // room for 256 in sample index though

// Minimum value to consider a sound, or maximum to be considered silent.
#define NOISE_FLOOR 0.0000001

static const double kSemiTone = exp2(1.0 / 12);

template <typename T, typename U>
void set_bit(T& dst, const U& bit, bool value) {
	if (value) {
		dst |= bit;
	} else {
		dst &= ~bit;
	}
}

double int_param(double value, double max) {
	return min(max, value * (max + 1));
}

enum TrigFlags {
	kTrigEnable = 1 << 0,
};

#if 0
int16 channel;			///< channel index in event bus
int16 pitch;			///< range [0, 127] = [C-2, G8] with A3=440Hz
float tuning;			///< 1.f = +1 cent, -1.f = -1 cent
float velocity;			///< range [0.0, 1.0]
int32 length;           ///< in sample frames (optional, Note Off has to follow in any case!)
int32 noteId;			///< note identifier (if not available then -1)
#endif
enum MidiTrigFlags {
	kMidiEnable = kTrigEnable,
	kMidiCC = 1 << 1,
};
// One whole pattern, but even that might be limiting?
#define MAX_NOTE_LENGTH 16
struct MidiTrig {
	uint8 flags;
	union {
		uint8 note;
		uint8 cc;
	};
	uint8 ccvalue;
	TQuarterNotes length;

	MidiTrig() : flags(0), ccvalue(0), length(0) {
		note = 0;
	}

	void set_param(uint8 type, double value) {
		switch (type) {
		case kParamMidiTrigEnable:
			set_bit(flags, kTrigEnable, value > 0.5);
			break;
		case kParamMidiTrigNote:
			note = int_param(value, 255);
			break;
		case kParamMidiTrigLength:
			length = int_param(value, MAX_NOTE_LENGTH);
		}
	}

	bool enabled() const {
		return !!(flags & kMidiEnable);
	}
	bool is_note() const {
		return !(flags & kMidiCC);
	}
};

struct DumbBuffer {
	const float *const buffer;
	const size_t length;

	DumbBuffer(const DumbBuffer&) = delete;
	DumbBuffer& operator=(const DumbBuffer&) = delete;
	DumbBuffer(const float *buffer, size_t length) : buffer(buffer), length(length) {}
	~DumbBuffer() {
		delete[] buffer;
	}

	static shared_ptr<DumbBuffer> copy_circular(const float *source, size_t position, size_t length) {
		float *output = new float[length];
		auto res = make_shared<DumbBuffer>(output, length);
		size_t n = length;
		while (n--) {
			if (++position == length) position = 0;
			*output++ = source[position];
		}
		return res;
	}

	float safe_get(double position, bool loop) {
		size_t p = loop ? fmod(position, length) : position;
		if (p < 0 || p >= length) {
			return 0;
		}
		return buffer[p];
	}
};

struct Playback {
	shared_ptr<DumbBuffer> source;
	double rate;
	double position;
	double level;
	bool loop;

	Playback() : rate(0), position(0), level(1.0), loop(false) {}

	bool fill(float *const dst, const int32 channel, const int32 count, const double trackLevel) const {
		double p = position;
		bool sound = false;
		assert(source);
		double f = level * trackLevel;
		for (int32 i = 0; i < count; i++) {
			float s = (dst[i] += source->safe_get(p, loop) * f);
			if (s > NOISE_FLOOR) {
				sound = true;
			}
			p += rate;
		}
		return sound;
	}

	// Returns true if finished.
	bool advance(const int32 count) {
		position += rate * count;
		if (loop) {
			position = fmod(position, source->length);
		} else if (position > source->length) {
			rate = 0;
			return true;
		}
		return false;
	}
};

enum SampleTrigFlags {
	kSampleEnable = kTrigEnable,
	kSampleStack = 1 << 1,
};
struct SampleTrig {
	uint8 flags;
	union {
		// If flags & kSampleStack = 1
		uint8 stack;
		// If flags & kSampleStack = 0
		uint8 sample;
	};
	// compared to original rate in sample, for now a stupid pitching thing, no interpolation
	double rate;
	double level;

	SampleTrig() : flags(0), stack(0), rate(1.0), level(1.0) {}

	bool enabled() const {
		return !!(flags & kSampleEnable);
	}

	void set_param(uint8 type, double value) {
		switch (type) {
		case kParamSampleTrigEnable:
			set_bit(flags, kSampleEnable, value > 0.5);
			break;
		/*case kParamSampleTrigOneShot:
			set_bit(flags, kSampleOneShot, value > 0.5);
			break;*/
		case kParamSampleTrigStack:
			set_bit(flags, kSampleStack, value > 0.5);
			break;
		case kParamSampleTrigSampleNumber:
			sample = int_param(value, 255);
			break;
		// case kParamSampleTrigRate:
		// case kParamSampleTrigLevel:
		}
	}
};

enum LatchTrigFlags {
	kLatchEnable = kTrigEnable,
	kLatchOneShot = 1 << 1,
};

struct LatchTrig {
	uint8 flags;
	uint8 source;
	uint8 sample;

	LatchTrig() : flags(0), source(0), sample(0) {}

	bool enabled() const {
		return !!(flags & kLatchEnable);
	}

	bool oneshot() const {
		return !!(flags & kLatchOneShot);
	}

	void set_param(uint8 type, double value) {
		switch (type) {
		case kParamLatchTrigEnable:
			set_bit(flags, kLatchEnable, value > 0.5);
			break;
		case kParamLatchTrigOneShot:
			set_bit(flags, kLatchOneShot, value > 0.5);
			break;
		case kParamLatchTrigSampleNumber:
			sample = int_param(value, 255);
			break;
		case kParamLatchTrigSource:
			source = int_param(value, 255);
			break;
		}
	}
};

struct SampleBuffer {
	float *buffer;
	size_t position, length;

	SampleBuffer() : buffer(NULL), position(0), length(0) {}
	~SampleBuffer() {
		delete[] buffer;
		buffer = NULL;
		position = length = 0;
	}

	void set_length(const size_t new_length) {
		if (new_length == length) return;
		float *new_buffer = new float[new_length];
		size_t new_position = 0;
		assert(new_buffer);
		if (new_length >= length) {
			size_t p = position;
			// copy [p, p) to [0,length), clear out [length, new_length)
			for (size_t n = length; n--;) {
				new_buffer[new_position++] = buffer[p++];
				if (p == length) p = 0;
			}
			memset(new_buffer + new_position, 0, (new_length - length) * sizeof(float));
		} else {
			// copy the most recent samples, [p - new_length, p) to [0, new_length)
			size_t p = position, n = new_length;
			while (n--) {
				new_buffer[n] = buffer[p];
				if (p > 0) {
					p--;
				} else {
					p = length;
				}
			}
		}
		delete[] buffer;
		buffer = new_buffer;
		position = new_position;
		length = new_buffer ? new_length : 0;
	}

	void add_samples(const float *input, size_t n) {
		if (!length) return;

		while (n--) {
			buffer[position++] = *input++;
			if (position == length) position = 0;
		}
	}

	shared_ptr<DumbBuffer> latch() const {
		return DumbBuffer::copy_circular(buffer, position, length);
	}
};

typedef std::deque<shared_ptr<DumbBuffer>> SampleStack;

struct InputChannel
{
	InputChannel() {
		std::fill_n(direct, NUM_OUTPUTS, 0.0f);
	}
	~InputChannel() {
	}
	
	SampleBuffer sampler;
	float direct[NUM_OUTPUTS];

	void set_length(const size_t new_length) {
		sampler.set_length(new_length);
	}

	void add_samples(const float *src, size_t count) {
		sampler.add_samples(src, count);
	}


	shared_ptr<DumbBuffer> latch() {
		return sampler.latch();
	}
};

struct Track {
	// 16 steps (for now)
	MidiTrig midi_trigs[PATTERN_LENGTH];
	SampleTrig sample_trigs[PATTERN_LENGTH];
	LatchTrig latch_trigs[PATTERN_LENGTH];
	Playback playback; // One playback per track, for now
	double level;
	bool armed;

	Track() : level(1.0), armed(false) {}
	~Track() {
	}

	void arm() {
		armed = true;
		// Maybe: set arm flag for each trigger.
	}
	void disarm() {
		armed = false;
	}
};

struct VTrackEffect : public AudioEffect {
	// 8 tracks of trigs
	Track tracks[NUM_TRACKS];
	// 4 input channels
	InputChannel input_channels[NUM_INPUTS];
	// Self samplers
	SampleBuffer output_samplers[NUM_OUTPUTS];
	// Saved samples
	vector<shared_ptr<DumbBuffer>> samples;

	double tempo;
	// pattern = 4 bars, 16 QNs, used when we don't receive any better position info
	TQuarterNotes positionInPattern;
	float last_vu;

	bool playing, recording;

	VTrackEffect() : tempo(0), positionInPattern(0), last_vu(0), playing(false), recording(false) {
		setControllerClass(VTrackControllerUID);
		set_tempo(120);
		for (int i = 0; i < NUM_INPUTS; i++) {
			for (int o = 0; o < NUM_OUTPUTS; o++) {
				input_channels[i].direct[o] = o < 2 ? 1 : 0;
			}
		}
		samples.resize(256);

		LatchTrig &latch = tracks[0].latch_trigs[0];
		latch.flags = kLatchEnable;
		//latch.flags |= kLatchOneShot;
		latch.source = 0;
		latch.sample = 0;
		SampleTrig &t = tracks[0].sample_trigs[0];
		t.flags = kSampleEnable;
		t.rate = pow(kSemiTone, 5);
		t.sample = 0;
		t.level = 1;
	}

	tresult PLUGIN_API initialize(FUnknown *context) {
		tresult result = AudioEffect::initialize(context);
		// if everything Ok, continue
		if (result != kResultOk) {
			return result;
		}

		addAudioInput(STR16("Stereo In A/B"), SpeakerArr::kStereo);
		addAudioInput(STR16("Stereo In C/D"), SpeakerArr::kStereo);
		addAudioOutput(STR16("Main Out"), SpeakerArr::kStereo);
		addAudioOutput(STR16("Cue Out"), SpeakerArr::kStereo, kAux, 0);

		addEventInput(STR16("Midi In"), 1);
		addEventOutput(STR16("Midi Out"), 1);

		return kResultOk;
	}

	tresult PLUGIN_API setupProcessing(ProcessSetup& newSetup) {
		newSetup.sampleRate = SAMPLE_RATE; // Support only exactly this sample rate. Hope this is a working way to tell the VST host about that :)
		newSetup.symbolicSampleSize = kSample32;
		return AudioEffect::setupProcessing(newSetup);
	}

	static bool hasState(ProcessContext *ctx, uint32 states) {
		return (ctx->state & states) == states;
	}

	void set_tempo(double new_tempo) {
		if (tempo != new_tempo) {
			tempo = new_tempo;
			update_sample_buffers();
		}
	}

	double samples_per_qn() {
		return 60 * SAMPLE_RATE / tempo;
	}

	double samples_per_trig() {
		return samples_per_qn() / TRIGS_PER_QN;
	}

	void update_sample_buffers() {
		// "bar"/"pattern" confusion here - but for now, one bar == one pattern
		size_t samplesPerBar = ceil(samples_per_qn() * PATTERN_LENGTH_QN);
		if (samplesPerBar != input_channels[0].sampler.length) {
			Debug("Tempo changed to %.1f BPM, %lu samples/bar\n", tempo, samplesPerBar);
		}
		for (int i = 0; i < NUM_INPUTS; i++) {
			input_channels[i].set_length(samplesPerBar);
		}
		for (int i = 0; i < NUM_OUTPUTS; i++) {
			output_samplers[i].set_length(samplesPerBar);
		}
	}

#define BAIL(...) do { Debug(__VA_ARGS__); return kInvalidArgument; } while (0)

	void process_trig_param(const ParamId id, double value) {
		Debug("Track %d trig %d: param %d => %g", id.track, id.trig, id.type, value);
		if (id.track == 0xff || id.trig == 0xff) { // Wildcard changes not handled.
			return;
		}
		if (id.latch_trig_related()) {
			tracks[id.track].latch_trigs[id.trig].set_param(id.type, value);
		} else if (id.midi_trig_related()) {
			tracks[id.track].midi_trigs[id.trig].set_param(id.type, value);
		} else {
			tracks[id.track].sample_trigs[id.trig].set_param(id.type, value);
		}
	}

	void process_parameter_queue(IParamValueQueue *vq) {
		int32 offsetSamples;
		int32 numPoints = vq->getPointCount();
		ParamID rawParamID = vq->getParameterId();
		double value;
		if (vq->getPoint(numPoints - 1, offsetSamples, value) != kResultTrue) {
			Debug("Invalid point for param %#x (%d points)\n", rawParamID, numPoints);
			return;
		}
		const ParamId id = ParamId(rawParamID);
		if (id.trig_related()) {
			process_trig_param(id, value);
			return;
		}
		switch (id.type) {
		case kParamArm: {// Perhaps better done as an Event?
			bool arm = value > 0.5;
			assert(id.trig == 0xff); // Only support global trig arming for now
			if (id.track == 0xff) {
				for (int t = 0; t < NUM_TRACKS; t++) {
					arm ? tracks[t].arm() : tracks[t].disarm();
				}
			} else if (arm) {
				tracks[id.track].arm();
			} else {
				tracks[id.track].disarm();
			}
			break;
		}
		default:
			Debug("Unhandled param %d (%#x value %g)\n", id.type, rawParamID, value);
		}
	}

	void process_trigs(ProcessData& data, TQuarterNotes time, int32 sampleOffset) {
		if (!playing) return;

		IEventList *output = data.outputEvents;
		int trig = fmod(time, PATTERN_LENGTH_QN) * TRIGS_PER_QN;
		for (int i = 0; i < NUM_TRACKS; i++) {
			Track& track = tracks[i];
			Event e;
			const MidiTrig& midi = track.midi_trigs[trig];
			if (midi.enabled() && midi.is_note()) {
				Debug("Midi trig: note %d length %.1f", midi.note, midi.length);
				e.type = Event::kNoteOnEvent;
				e.sampleOffset = sampleOffset;
				e.ppqPosition = time;
				e.busIndex = 0;
				e.noteOn.channel = i;
				e.noteOn.pitch = midi.note;
				e.noteOn.length = midi.length * samples_per_qn();
				e.noteOn.noteId = -1;
				output->addEvent(e);
				e.type = Event::kNoteOffEvent;
				e.sampleOffset += e.noteOn.length;
				e.ppqPosition += midi.length;
				e.noteOff.channel = i;
				e.noteOff.pitch = midi.note;
				e.noteOff.noteId = -1;
				output->addEvent(e);
			}
			LatchTrig &latch = track.latch_trigs[trig];
			if (latch.enabled() && (track.armed || !latch.oneshot())) {
				track.armed = false;
				// TODO: latch.armed = false?
				Debug("Latch trig %d @%.1f: input channel %d, oneshot=%d\n", trig, time, latch.source, latch.oneshot());
				// if (!(latch.flags & kLatchSampleOutput))
				samples[latch.sample] = input_channels[latch.source].latch();
				//Debug("Channel %d: now %u stacked\n", i, (unsigned)chan.sample_stack.size());
			}
			const SampleTrig& sample = track.sample_trigs[trig];
			if (sample.enabled()) {
				Playback& playback = track.playback;
				if (sample.flags & kSampleStack) {
					assert(!"stack samples not implemented anymore");
#if 0
					uint8 input = sample.stack / MAX_STACK_SIZE;
					uint8 stack = sample.stack % MAX_STACK_SIZE;
					Debug("Sample trig %d @%.1f: input %d/stack %d rate %.1fs\n", trig, time, input, stack, sample.rate);
					// TODO Move the sample stack from InputChannel to Track, let Latch trigs include source channel.
					if (input >= NUM_INPUTS) {
						Debug("Invalid input %d >= %d\n", input, NUM_INPUTS);
						continue;
					}
					const SampleStack& sample_stack = input_channels[input].sample_stack;
					if (stack >= sample_stack.size()) {
						Debug("Invalid stack index %d >= %d\n", stack, sample_stack.size());
						continue;
					}
					playback.source = sample_stack[stack];
#endif
				} else {
					Debug("Sample trig %d @%.1f: sample %d rate=%.3fx level=%.1f\n", trig, time, sample.sample, sample.rate, sample.level);
					playback.source = samples[sample.sample];
				}
				if (playback.source) {
					playback.rate = sample.rate;
					playback.level = sample.level;
					playback.position = 0;
				}
			}
		}
	}

	tresult PLUGIN_API process(ProcessData& data) {
		if (IParameterChanges *params = data.inputParameterChanges) {
			const int32 n = params->getParameterCount();
			for (int32 i = n; i < n; i++) {
				process_parameter_queue(params->getParameterData(i));
			}
		}
		if (data.processContext) {
			update_context(data.processContext);
		}
		copy_events(data);
		reset_silence(data.outputs, data.numOutputs, data.numSamples);
		int32 sample_position = 0;
		// TODO Use full song position here (it would work since we use fmod, but it will make debug printouts more useful, and give some insight into how song position and the host's sequencer works).
		TQuarterNotes musicTime = positionInPattern;
		while (sample_position < data.numSamples) {
			int32 next_event = process_events(data, sample_position);
			int32 next_trig = get_next_trig(musicTime, sample_position);
			if (next_trig == sample_position) {
				process_trigs(data, musicTime, sample_position);
				next_trig += samples_per_trig();
			}
			int32 next_sample_pos = min(data.numSamples, next_trig);
			if (next_event >= 0) {
				next_sample_pos = min(next_sample_pos, next_event);
			}
			int32 num_samples = next_sample_pos - sample_position;
			assert(next_sample_pos <= data.numSamples);
			if (num_samples) {
				process(data, sample_position, num_samples);
			}

			musicTime += num_samples / samples_per_qn();
			if (musicTime > PATTERN_LENGTH_QN) musicTime -= PATTERN_LENGTH_QN;
			sample_position = next_sample_pos;
		}
		positionInPattern = musicTime;

		float in_vu = get_vu(data.inputs, data.numInputs, data.numSamples);
		float out_vu = get_vu(data.outputs, data.numOutputs, data.numSamples);
		float vu = out_vu;
		IParameterChanges* paramChanges = data.outputParameterChanges;
		if (paramChanges && last_vu != vu) {
			int32 index = 0;
			IParamValueQueue* paramQueue = paramChanges->addParameterData(kVuPPMId, index);
			if (paramQueue) {
				int32 index2 = 0;
				paramQueue->addPoint(0, vu, index2);
			}
			last_vu = vu;
		}

		return kResultOk;
	}

	int32 get_next_trig(TQuarterNotes qn, int32 samples) {
		TQuarterNotes next_trig_qn = ceil(qn * TRIGS_PER_QN) / TRIGS_PER_QN;
		return samples + (next_trig_qn - qn) * samples_per_qn();
	}

	int32 get_next_qn(TQuarterNotes qn, int32 samples) {
		TQuarterNotes next_qn = ceil(qn);
		return samples + (next_qn - qn) * samples_per_qn();
	}

	void set_position(TQuarterNotes barPosition, TQuarterNotes projectTime) {
		// Last bar was at (project time) barPosition. We don't really care here but just use the project time directly.
		positionInPattern = fmod(projectTime, PATTERN_LENGTH_QN);
	}

	void update_context(ProcessContext* ctx) {
		if (hasState(ctx, ProcessContext::kBarPositionValid | ProcessContext::kProjectTimeMusicValid)) {
			set_position(ctx->barPositionMusic, ctx->projectTimeMusic);
		}
		if (hasState(ctx, ProcessContext::kTempoValid)) {
			set_tempo(ctx->tempo);
		}
		playing = hasState(ctx, ProcessContext::kPlaying);
		recording = hasState(ctx, ProcessContext::kRecording);
	}

	void copy_events(ProcessData& data) {
		if (IEventList *events = data.inputEvents) {
			int32 n = events->getEventCount(), i = 0;
			if (!n) return;
			IEventList *out = data.outputEvents;
			Event e;
			while (n--) {
				// e.sampleOffset and e.ppqPosition
				events->getEvent(i++, e);
				if (e.type != Event::kDataEvent) {
					Debug("Event @%.1f (%d) on bus %d: type %d\n", e.ppqPosition, e.sampleOffset, e.busIndex, e.type);
				}
				out->addEvent(e);
			}
		}
	}

	// Returns the sample position of the next event to process
	int32 process_events(ProcessData& data, const int32 sample_position) {
		if (IEventList *events = data.inputEvents) {
			const int32 n = events->getEventCount();
			Event e;
			for (int32 i = 0; i < n; i++) {
				events->getEvent(i, e);
				if (e.sampleOffset < sample_position) {
					continue;
				} else if (e.sampleOffset > sample_position) {
					return e.sampleOffset;
				}
				// TODO Actually process events :)
			}
		}
		return -1;
	}

	void reset_silence(AudioBusBuffers* outputs, int32 numOutputs, int32 numSamples) {
		for (int32 bus = 0; bus < numOutputs; bus++) {
			reset_silence(outputs[bus], numSamples);
		}
	}

	void reset_silence(AudioBusBuffers& output, const int32 numSamples) {
		output.silenceFlags = (1 << output.numChannels) - 1;
		for (int32 c = 0; c < output.numChannels; c++) {
			memset(output.channelBuffers32[c], 0, numSamples * sizeof(float));
		}
	}

	float get_vu(AudioBusBuffers *buses, size_t numBuses, size_t numSamples) {
		float vu = 0;
		for (int32 bus = 0; bus < numBuses; bus++) {
			AudioBusBuffers &outp = buses[bus];
			for (int32 c = 0; c < outp.numChannels; c++) {
				float *src = outp.channelBuffers32[c];
				for (int32 s = 0; s < numSamples; s++) {
					vu = max(vu, src[s]);
				}
			}
		}
		return vu;
	}

	void process(ProcessData& data, const int32 offset, const int32 count) {
		process_inputs(data, offset, count);
		process_samples(data, offset, count);
		process_outputs(data, offset, count);
	}

	void process_samples(ProcessData& data, const int32 offset, const int32 count) {
		AudioBusBuffers &outp = data.outputs[0];
		for (int i = 0; i < NUM_TRACKS; i++) {
			Playback& playback = tracks[i].playback;
			if (!playback.rate || !playback.level || !tracks[i].level) continue;

			for (int32 c = 0; c < 2; c++) {
				float *dst = outp.channelBuffers32[c] + offset;
				if (playback.fill(dst, c, count, tracks[i].level)) {
					outp.silenceFlags &= ~(1ull << c);
				}
			}
			if (playback.advance(count)) {
				Debug("Playback finished on track %d\n", i);
			}
		}
	}

	void process_inputs(ProcessData& data, const int32 offset, const int32 count) {
		assert(data.numOutputs >= 1);
		assert(count > 0);
		assert(offset < data.numSamples);
		assert(offset + count > offset && offset + count <= data.numSamples);

		AudioBusBuffers &outp = data.outputs[0];
		size_t input_channel_index = 0;
		for (int32 bus = 0; bus < data.numInputs; bus++) {
			AudioBusBuffers &inp = data.inputs[bus];
			assert(outp.numChannels == 2);
			for (int32 c = 0; c < inp.numChannels; c++) {
				InputChannel& chan = input_channels[input_channel_index];
				const float *const src = inp.channelBuffers32[c] + offset;
				chan.add_samples(src, count);
				for (int32 outc = 0; outc < 2; outc++) {
					float *const dst = outp.channelBuffers32[outc] + offset;
					bool silent = !!(inp.silenceFlags & (1ull << c));
					float f = chan.direct[outc];
					if (silent || abs(f) <= NOISE_FLOOR) {
						// count > 0, so max_element will return something.
						silent = *std::max_element(dst, dst + count) > NOISE_FLOOR;
					} else {
						silent = true;
						for (int32 s = 0; s < count; s++) {
							float val = (dst[s] += f * src[s]);
							if (val > NOISE_FLOOR) {
								silent = false;
							}
						}
					}
					// output channels are set silent by default, then cleared whenever we output a non-zero sample on them.
					if (!silent) {
						outp.silenceFlags &= ~(1ull << outc);
					}
				}
				input_channel_index++;
			}
		}
	}

	void process_outputs(ProcessData& data, const int32 offset, const int32 count) {
		assert(data.numOutputs >= 1);
		size_t output_channel_index = 0;
		for (int32 bus = 0; bus < data.numOutputs; bus++) {
			AudioBusBuffers &outp = data.outputs[bus];
			for (int32 c = 0; c < outp.numChannels; c++) {
				output_samplers[output_channel_index++].add_samples(outp.channelBuffers32[c] + offset, count);
			}
		}
	}

tresult PLUGIN_API terminate()
{
	HERE;
	return AudioEffect::terminate();
}


~VTrackEffect()
{
	HERE;
}

};

FUnknown *createVTrackEffect(void *context) {
	return (IAudioProcessor*)new VTrackEffect();
}
