class ReceiverPCM extends AudioWorkletProcessor {
  constructor(options) {
    super();this.frames=options.processorOptions.segmentFrames;
    this.buffer=new Float32Array(this.frames);this.offset=0;
  }
  process(inputs) {
    const channels=inputs[0];
    if (!channels || !channels.length) return true;
    for(let i=0;i<channels[0].length;i++) {
      let value=0;for(const channel of channels)value+=channel[i];
      this.buffer[this.offset++]=value/channels.length;
      if(this.offset===this.frames) {
        const buffer=this.buffer.buffer;this.port.postMessage(buffer,[buffer]);
        this.buffer=new Float32Array(this.frames);this.offset=0;
      }
    }
    return true;
  }
}
registerProcessor('receiver-pcm',ReceiverPCM);
