
# import the env from pip
import LiveStreamingEnv.env as env
import LiveStreamingEnv.load_trace as load_trace
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import tensorflow as tf
import a3c
# path setting
TRAIN_TRACES = './network_trace/'   #train trace path setting,
video_size_file = './new_video_trace/video/game_1/frame_trace_'      #video trace path setting,
LogFile_Path = "./log/"                #log file trace path setting,

DEBUG = False
DRAW = False
# load the trace
all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)
#random_seed 
random_seed = 2
video_count = 0
FPS = 25
frame_time_len = 0.04
#init the environment
#setting one:
#     1,all_cooked_time : timestamp
#     2,all_cooked_bw   : throughput
#     3,all_cooked_rtt  : rtt
#     4,agent_id        : random_seed
#     5,logfile_path    : logfile_path
#     6,VIDEO_SIZE_FILE : Video Size File Path
#     7,Debug Setting   : Debug
net_env = env.Environment(all_cooked_time=all_cooked_time,
			  all_cooked_bw=all_cooked_bw,
			  random_seed=random_seed,
			  logfile_path=LogFile_Path,
			  VIDEO_SIZE_FILE=video_size_file,
			  Debug = DEBUG)

BIT_RATE      = [500.0,850.0,1200.0,1850.0] # kpbs
TARGET_BUFFER = [2.0,3.0]   # seconds
# ABR setting
RESEVOIR = 0.5
CUSHION  = 2

cnt = 0
# defalut setting
last_bit_rate = 0
bit_rate = 0
target_buffer = 0
# QOE setting
reward_frame = 0
reward_all = 0
SMOOTH_PENALTY= 0.02 
REBUF_PENALTY = 1.5 
LANTENCY_PENALTY = 0.005 

# plot info
idx = 0
id_list = []
bit_rate_record = []
buffer_record = []
throughput_record = []
# plot the real time image
if DRAW:
    fig = plt.figure()
    plt.ion()
    plt.xlabel("time")
    plt.axis('off')


# my param
S_INFO=7
S_LEN=8
A_DIM=4
ACTOR_LR_RATE=0.0001
CRITIC_LR_RATE=0.01
TRAIN_SEQ_LEN = 200  # take as a train batch
MODEL_SAVE_INTERVAL = 10
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
GRADIENT_BATCH_SIZE = 16
RANDOM_SEED=42
SUMMARY_DIR='./results'
LOG_FILE = './results/log'
NN_MODEL = None
DEFAULT_QUALITY=1
RAND_RANGE = 1000000

if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

with tf.Session() as sess, open(LOG_FILE, 'w') as log_file:

    actor = a3c.ActorNetwork(sess,
                                state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

    critic = a3c.CriticNetwork(sess,
                                state_dim=[S_INFO, S_LEN],
                                learning_rate=CRITIC_LR_RATE)

    summary_ops, summary_vars = a3c.build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
    saver = tf.train.Saver()  # save neural net parameters

    # restore neural net parameters
    nn_model = NN_MODEL
    if nn_model is not None:  # nn_model is the path to file
        saver.restore(sess, nn_model)
        print("Model restored.")

    epoch = 0
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []

    actor_gradient_batch = []
    critic_gradient_batch = []

    while True:
            reward_frame = 0
            # input the train steps
            #if cnt > 5000:
                #plt.ioff()
            #    break
            #actions bit_rate  target_buffer
            # every steps to call the environment
            # time           : physical time 
            # time_interval  : time duration in this step
            # send_data_size : download frame data size in this step
            # chunk_len      : frame time len
            # rebuf          : rebuf time in this step          
            # buffer_size    : current client buffer_size in this step          
            # rtt            : current buffer  in this step          
            # play_time_len  : played time len  in this step          
            # end_delay      : end to end latency which means the (upload end timestamp - play end timestamp)
            # decision_flag  : Only in decision_flag is True ,you can choose the new actions, other time can't Becasuse the Gop is consist by the I frame and P frame. Only in I frame you can skip your frame
            # buffer_flag    : If the True which means the video is rebuffing , client buffer is rebuffing, no play the video
            # cdn_flag       : If the True cdn has no frame to get 
            # end_of_video   : If the True ,which means the video is over.
            time, time_interval, send_data_size, chunk_len,\
                    rebuf, buffer_size, play_time_len,end_delay,\
                    cdn_newest_id, download_id,cdn_has_frame,decision_flag, \
                    buffer_flag,cdn_flag, end_of_video = net_env.get_video_frame(bit_rate,target_buffer)
            cnt += 1
            # print(time_interval,end_delay)
            
            '''if time_interval != 0:
                # plot bit_rate 
                id_list.append(idx)
                idx += time_interval
                bit_rate_record.append(BIT_RATE[bit_rate])
                # plot buffer 
                buffer_record.append(buffer_size)
                # plot throughput 
                trace_idx = net_env.get_trace_id()
                print(trace_idx, idx,len(all_cooked_bw[trace_idx]))
                throughput_record.append(all_cooked_bw[trace_idx][int(idx/0.5)] * 1000 )'''
            if not cdn_flag:
                reward_frame = frame_time_len * float(BIT_RATE[bit_rate]) / 1000  - REBUF_PENALTY * rebuf - LANTENCY_PENALTY * end_delay
            else:
                reward_frame = -(REBUF_PENALTY * rebuf)
            if decision_flag or end_of_video:
                # reward formate = play_time * BIT_RATE - 4.3 * rebuf - 1.2 * end_delay
                reward_frame += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
                # last_bit_rate
                last_bit_rate = bit_rate

                # draw setting
                if DRAW:
                    ax = fig.add_subplot(311)
                    plt.ylabel("BIT_RATE")
                    plt.ylim(300, 1000)
                    plt.plot(id_list, bit_rate_record, '-r')

                    ax = fig.add_subplot(312)
                    plt.ylabel("Buffer_size")
                    plt.ylim(0, 7)
                    plt.plot(id_list, buffer_record, '-b')

                    ax = fig.add_subplot(313)
                    plt.ylabel("throughput")
                    plt.ylim(0, 2500)
                    plt.plot(id_list, throughput_record, '-g')

                    plt.draw()
                    plt.pause(0.01)

            if(time_interval<0.00001):
                continue

            

            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)
            if decision_flag or end_of_video:
                r_batch.append(reward_frame)
                state = np.roll(state, -1, axis=1)

                state[0, -1] = BIT_RATE[bit_rate] / float(np.max(BIT_RATE))  # last quality
                state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
                state[2, -1] = float(send_data_size) / float(time_interval) / M_IN_K  # kilo bit / ms
                state[3, -1] = float(end_delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec 
                state[4, :A_DIM] = np.array(send_data_size)/1000000.
                state[5, -1] = cdn_newest_id - download_id/100.                
                state[6, -1] = rebuf/10.

                action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
                action_cumsum = np.cumsum(action_prob)
                r=np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)
                bit_rate = (action_cumsum > r).argmax()

                # print(bit_rate)

                entropy_record.append(a3c.compute_entropy(action_prob[0]))

                log_file.write(str(time) + '\t' +
                                str(BIT_RATE[bit_rate]) + '\t' +
                                str(buffer_size) + '\t' +
                                str(rebuf) + '\t' +
                                str(send_data_size) + '\t' +
                                str(end_delay) + '\t' +
                                str(reward_frame) + '\n')
                log_file.flush()

                if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                    actor_gradient, critic_gradient, td_batch = \
                        a3c.compute_gradients(s_batch=np.stack(s_batch[1:], axis=0),  # ignore the first chuck
                                                a_batch=np.vstack(a_batch[1:]),  # since we don't have the
                                                r_batch=np.vstack(r_batch[1:]),  # control over it
                                                terminal=end_of_video, actor=actor, critic=critic)
                    td_loss = np.mean(td_batch)

                    actor_gradient_batch.append(actor_gradient)
                    critic_gradient_batch.append(critic_gradient)

                    print ("====")
                    print ("Time", epoch)
                    print ("buffer", td_loss, "Throughput", np.mean(r_batch), "reward", np.mean(entropy_record))
                    print ("====")

                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: td_loss,
                        summary_vars[1]: np.mean(r_batch),
                        summary_vars[2]: np.mean(entropy_record)
                    })

                    writer.add_summary(summary_str, epoch)
                    writer.flush()

                    entropy_record = []

                    if len(actor_gradient_batch) >= GRADIENT_BATCH_SIZE:

                        assert len(actor_gradient_batch) == len(critic_gradient_batch)

                        for i in range(len(actor_gradient_batch)):
                            actor.apply_gradients(actor_gradient_batch[i])
                            critic.apply_gradients(critic_gradient_batch[i])

                        actor_gradient_batch = []
                        critic_gradient_batch = []

                        epoch += 1
                        if epoch % MODEL_SAVE_INTERVAL == 0:
                            # Save the neural net parameters to disk.
                            save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                                    str(epoch) + ".ckpt")
                            print("Model saved in file: %s" % save_path)

                    del s_batch[:]
                    del a_batch[:]
                    del r_batch[:]

                if end_of_video:
                    last_bit_rate = DEFAULT_QUALITY
                    bit_rate = DEFAULT_QUALITY  # use the default action here

                    action_vec = np.zeros(A_DIM)
                    action_vec[bit_rate] = 1

                    s_batch.append(np.zeros((S_INFO, S_LEN)))
                    a_batch.append(action_vec)

                else:
                    s_batch.append(state)

                    action_vec = np.zeros(A_DIM)
                    action_vec[bit_rate] = 1
                    a_batch.append(action_vec)

                if rebuf<0.00001:
                    target_buffer=0
                else:
                    target_buffer=1
            # ------------------------------------------- End  ------------------------------------------- 

            reward_all += reward_frame
            if end_of_video:
                # Narrow the range of results
                break
                
    if DRAW:
        plt.show()
    print(reward_all)
