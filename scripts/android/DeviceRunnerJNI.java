package com.google.ai.edge.litertlm;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

/** Three-line request protocol shared with device_runner.c, using SDK 0.17.0. */
public final class DeviceRunnerJNI {
  private static double seconds(long start) { return (System.nanoTime()-start)/1e9; }
  private static double finite(double value) { return Double.isFinite(value) ? value : -1.0; }
  public static void main(String[] args) throws Exception {
    if (args.length < 3) throw new IllegalArgumentException("MODEL BACKEND CACHE_DIR [NPU_LIB_DIR]");
    long begin=System.nanoTime();
    LiteRtLmJni api=LiteRtLmJni.INSTANCE;
    String backend=args[1].toUpperCase(java.util.Locale.ROOT);
    long engine=api.nativeCreateEngine(args[0],backend,"","",-1,-1,args[2],true,
        Boolean.FALSE,args.length>3?args[3]:"","","",backend.equals("CPU")?4:-1,-1,-1);
    JSONObject ready=new JSONObject();
    ready.put("pid",android.os.Process.myPid());ready.put("backend",args[1]);
    ready.put("runtime","litertlm-android-0.17.0");ready.put("init_seconds",seconds(begin));
    System.out.println("READY "+ready);
    int index=0;
    try {
      BufferedReader input=new BufferedReader(new InputStreamReader(System.in,StandardCharsets.UTF_8));
      String limits;
      while ((limits=input.readLine())!=null) {
        String[] pieces=limits.trim().split("\\s+");
        if (pieces.length!=2) throw new IllegalArgumentException("Invalid limits line");
        int max=Integer.parseInt(pieces[0]);boolean thinking=Integer.parseInt(pieces[1])!=0;
        if (max<1 || max>32768) throw new IllegalArgumentException("Invalid token ceiling");
        String history=input.readLine(),message=input.readLine();
        if (history==null || message==null) throw new IllegalArgumentException("Truncated request");
        String extra="{\"enable_thinking\":"+thinking+"}";
        SamplerConfig sampler=new SamplerConfig(1,1.0,0.0,42);
        ThinkingConfig tc=new ThinkingConfig(thinking,-1);
        long conversation=api.nativeCreateConversation(engine,sampler,history,"[]",null,extra,
            false,null,null,null,null,false,max,tc,false);
        try {
          long started=System.nanoTime();
          String raw=api.nativeSendMessage(conversation,message,extra,null,null,null,null,max,tc,0,null);
          BenchmarkInfo info=api.nativeConversationGetBenchmarkInfo(conversation);
          JSONObject result=new JSONObject();
          result.put("index",index++);result.put("elapsed_seconds",seconds(started));
          result.put("input_tokens",info.getLastPrefillTokenCount());
          result.put("output_tokens",info.getLastDecodeTokenCount());
          result.put("ttft_seconds",finite(info.getTimeToFirstTokenInSecond()));
          result.put("prefill_tps",finite(info.getLastPrefillTokensPerSecond()));
          result.put("decode_tps",finite(info.getLastDecodeTokensPerSecond()));
          result.put("response",new JSONObject(raw));
          System.out.println("RESULT "+result);
        } finally { api.nativeDeleteConversation(conversation); }
      }
    } finally { api.nativeDeleteEngine(engine); }
    System.err.println("RUNNER_DONE requests="+index);
  }
}
