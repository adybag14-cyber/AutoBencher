package dev.autobencher;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

/** Exercise the installed Gallery classes/native library without replacing the app. */
public final class DeviceRunnerReflect {
  private static Object call(Object object, String name, Object... args) throws Exception {
    Method selected = null;
    for (Method m : object.getClass().getMethods()) {
      if (m.getName().equals(name) && m.getParameterCount()==args.length) {
        if (selected!=null) throw new IllegalStateException("Ambiguous runtime method: "+name);
        selected=m;
      }
    }
    if (selected==null) throw new NoSuchMethodException(name+"/"+args.length);
    return selected.invoke(object,args);
  }
  private static Object make(String name, Class<?>[] types, Object... args) throws Exception {
    return Class.forName("com.google.ai.edge.litertlm."+name).getConstructor(types).newInstance(args);
  }
  private static double seconds(long start) { return (System.nanoTime()-start)/1e9; }
  private static Number metric(Object info,String name) throws Exception { return (Number)call(info,name); }
  public static void main(String[] args) throws Exception {
    if (args.length<3) throw new IllegalArgumentException("MODEL BACKEND CACHE_DIR [DISPATCH_DIR]");
    long start=System.nanoTime();
    Object api=Class.forName("com.google.ai.edge.litertlm.LiteRtLmJni").getField("INSTANCE").get(null);
    String backend=args[1].toUpperCase(java.util.Locale.ROOT);
    long engine=(Long)call(api,"nativeCreateEngine",args[0],backend,"","",-1,-1,args[2],true,
        Boolean.FALSE,args.length>3?args[3]:"","","",backend.equals("CPU")?4:-1,-1);
    JSONObject ready=new JSONObject();
    ready.put("pid",android.os.Process.myPid());ready.put("backend",args[1]);
    ready.put("runtime","installed-gallery-jni-14-argument-engine");ready.put("init_seconds",seconds(start));
    System.out.println("READY "+ready);
    int index=0;
    try {
      BufferedReader input=new BufferedReader(new InputStreamReader(System.in,StandardCharsets.UTF_8));
      String line;
      while ((line=input.readLine())!=null) {
        String[] limits=line.trim().split("\\s+");
        if (limits.length!=2) throw new IllegalArgumentException("Invalid limits record");
        int max=Integer.parseInt(limits[0]),mode=Integer.parseInt(limits[1]);
        if (max<1 || max>32768 || mode<0 || mode>1) throw new IllegalArgumentException("Invalid request limits");
        boolean thinking=mode==1;
        String history=input.readLine(),message=input.readLine();
        if (history==null || message==null) throw new IllegalArgumentException("Truncated request");
        String extra="{\"enable_thinking\":"+thinking+"}";
        Object sampler=make("SamplerConfig",new Class<?>[]{int.class,double.class,double.class,int.class},1,1.0,0.0,42);
        Object tc=make("ThinkingConfig",new Class<?>[]{boolean.class,int.class},thinking,-1);
        long conversation=(Long)call(api,"nativeCreateConversation",engine,sampler,history,"[]",null,extra,
            false,null,null,null,null,false,max,tc,false);
        try {
          long begin=System.nanoTime();
          String raw=(String)call(api,"nativeSendMessage",conversation,message,extra,null,null,null,null,max,tc,0,null);
          Object info=call(api,"nativeConversationGetBenchmarkInfo",conversation);
          JSONObject result=new JSONObject();
          result.put("index",index++);result.put("elapsed_seconds",seconds(begin));
          result.put("input_tokens",metric(info,"getLastPrefillTokenCount"));
          result.put("output_tokens",metric(info,"getLastDecodeTokenCount"));
          result.put("ttft_seconds",metric(info,"getTimeToFirstTokenInSecond"));
          result.put("prefill_tps",metric(info,"getLastPrefillTokensPerSecond"));
          result.put("decode_tps",metric(info,"getLastDecodeTokensPerSecond"));
          result.put("response",new JSONObject(raw));
          System.out.println("RESULT "+result);
        } finally { call(api,"nativeDeleteConversation",conversation); }
      }
    } finally { call(api,"nativeDeleteEngine",engine); }
    System.err.println("RUNNER_DONE requests="+index);
  }
}
