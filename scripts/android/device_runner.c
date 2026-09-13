#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "engine.h"
#include "conversation.h"

static double now_sec(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

// Each request is three newline-terminated records: max_output_tokens and
// thinking (0/1); JSON prior messages; JSON final user message. JSON strings must
// be serialized on one line. One fresh conversation is created per request.
int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: device_runner MODEL BACKEND CACHE_DIR [DISPATCH_DIR]\n");
    return 2;
  }
  setvbuf(stdout, NULL, _IOLBF, 0);
  setvbuf(stderr, NULL, _IOLBF, 0);
  const double started = now_sec();
  fprintf(stderr, "RUNNER_START pid=%ld model=%s backend=%s\n", (long)getpid(), argv[1], argv[2]);
  LiteRtLmEngineSettings *settings = litert_lm_engine_settings_create(argv[1], argv[2], NULL, NULL);
  if (!settings) return 3;
  litert_lm_engine_settings_set_cache_dir(settings, argv[3]);
  if (strcmp(argv[2], "cpu") == 0) litert_lm_engine_settings_set_num_threads(settings, 4);
  litert_lm_engine_settings_enable_benchmark(settings);
  if (argc > 4) litert_lm_engine_settings_set_litert_dispatch_lib_dir(settings, argv[4]);
  LiteRtLmEngine *engine = litert_lm_engine_create(settings);
  litert_lm_engine_settings_delete(settings);
  if (!engine) { fprintf(stderr, "ENGINE_CREATE_FAILED\n"); return 4; }
  printf("READY {\"pid\":%ld,\"backend\":\"%s\",\"init_seconds\":%.6f}\n", (long)getpid(), argv[2], now_sec()-started);
  char *limits = NULL, *history = NULL, *message = NULL;
  size_t limits_cap=0, history_cap=0, message_cap=0;
  unsigned index=0;
  int status=0;
  while (getline(&limits, &limits_cap, stdin) > 0) {
    int max_output=0, thinking=0;
    if (sscanf(limits, "%d %d", &max_output, &thinking) != 2 || max_output < 1 || max_output > 32768 || thinking < 0 || thinking > 1) { status=5; break; }
    if (getline(&history, &history_cap, stdin) <= 0 || getline(&message, &message_cap, stdin) <= 0) { status=6; break; }
    LiteRtLmSessionConfig *session=litert_lm_session_config_create();
    // The released C API advertises GREEDY and TOP_K, but the CPU factory
    // implements TOP_P. TOP_P with k=1,p=1,temp=0 is deterministic argmax on
    // both CPU and GPU and matches the Kotlin SDK's sampler configuration.
    LiteRtLmSamplerParams *sampler=litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP);
    litert_lm_sampler_params_set_top_k(sampler,1);
    litert_lm_sampler_params_set_top_p(sampler,1.0f);
    litert_lm_sampler_params_set_temperature(sampler,0.0f);
    litert_lm_sampler_params_set_seed(sampler,42);
    litert_lm_session_config_set_sampler_params(session,sampler);
    litert_lm_sampler_params_delete(sampler);
    litert_lm_session_config_set_max_output_tokens(session,max_output);
    LiteRtLmConversationConfig *config=litert_lm_conversation_config_create();
    litert_lm_conversation_config_set_session_config(config,session);
    litert_lm_session_config_delete(session);
    LiteRtLmThinkingConfig *tc=litert_lm_thinking_config_create();
    litert_lm_thinking_config_set_enable_thinking(tc,thinking != 0);
    litert_lm_conversation_config_set_thinking_config(config,tc);
    litert_lm_thinking_config_delete(tc);
    litert_lm_conversation_config_set_extra_context(config,thinking ? "{\"enable_thinking\":true}" : "{\"enable_thinking\":false}");
    litert_lm_conversation_config_set_messages(config,history);
    LiteRtLmConversation *conversation=litert_lm_conversation_create(engine,config);
    litert_lm_conversation_config_delete(config);
    if (!conversation) { fprintf(stderr,"CONVERSATION_CREATE_FAILED index=%u\n",index);status=7;break; }
    double begin=now_sec();
    LiteRtLmJsonResponse *response=litert_lm_conversation_send_message(conversation,message,NULL,NULL);
    if (!response) { fprintf(stderr,"GENERATION_FAILED index=%u\n",index);litert_lm_conversation_delete(conversation);status=8;break; }
    LiteRtLmBenchmarkInfo *bench=litert_lm_conversation_get_benchmark_info(conversation);
    int input=-1,output=-1;double ttft=-1,prefill=-1,decode=-1;
    if (bench) {
      input=litert_lm_benchmark_info_get_prefill_token_count_at(bench,0);
      output=litert_lm_benchmark_info_get_decode_token_count_at(bench,0);
      ttft=litert_lm_benchmark_info_get_time_to_first_token(bench);
      prefill=litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(bench,0);
      decode=litert_lm_benchmark_info_get_decode_tokens_per_sec_at(bench,0);
      litert_lm_benchmark_info_delete(bench);
    }
    printf("RESULT {\"index\":%u,\"elapsed_seconds\":%.6f,\"input_tokens\":%d,\"output_tokens\":%d,\"ttft_seconds\":%.6f,\"prefill_tps\":%.6f,\"decode_tps\":%.6f,\"response\":%s}\n",index++,now_sec()-begin,input,output,ttft,prefill,decode,litert_lm_json_response_get_string(response));
    litert_lm_json_response_delete(response);
    litert_lm_conversation_delete(conversation);
  }
  free(limits);free(history);free(message);
  litert_lm_engine_delete(engine);
  fprintf(stderr,"RUNNER_DONE requests=%u status=%d\n",index,status);
  return status;
}
