warning('off')
clear all
disubj = dir('/data/data_emg/R*');  % Set this to the data path
load('/data/data_emg/CURSOR_clean_Nov2023.mat')
conds = [];
for subj = 1:length(disubj)
    di = dir(['/data/data_emg/' disubj(subj).name filesep 'rml*']);
    for file = 1:length(di)
        load(['/data/data_emg/' disubj(subj).name filesep di(file).name]) % Set this to the data path
        clear colorFB
        if block > 2
            if any(reshape(all(Color_target_alltrials==[.8302 .5 1],2),[size(Color_target_alltrials,1)*size(Color_target_alltrials,3) 1]))
                colorKO = [.8302 .5 1];
                colorOK = [.5 .9495 1];
                % reinforced(file,subj) = false;
                reinforced = false;
            elseif any(reshape(all(Color_target_alltrials==[1 .5 .5],2),[size(Color_target_alltrials,1)*size(Color_target_alltrials,3) 1]))
                colorKO = [1 .5 .5];
                colorOK = [.5 1 .5];
                % reinforced(file,subj) = true;
                reinforced = true;
            else
                colorOK = [.6 .6 .6];
                colorKO = [.6 .6 .6];
                % reinforced(file,subj) = NaN;
                reinforced = NaN;
            end
            if ~isnan(reinforced)
                if length(disubj(subj).name)==6
                    sub_idx = length(disubj(subj).name(end));
                else
                    sub_idx = length(disubj(subj).name(end-1:end));
                end

                colorFB = squeeze(all(Color_target_alltrials==colorOK,2));
                % T=table(CURSOR(61:420,:),colorFB(61:420,:),repmat(Seq_target(61:420),1,29),'VariableNames',{'cursor','colour','target'});
                T=table(CURSOR_allsubjects(61:end,:, block, sub_idx),colorFB(61:420,:),repmat(Seq_target(61:end),1,29),'VariableNames',{'cursor','colour','target'});


                if block == 10
                    block_ = 0;
                else
                    block_=block;
                end

                %writetable(T,['/data/preprocessed_emg/' disubj(subj).name '_block' num2str(block) '_cond' num2str(cond) '.csv'])
                writetable(T,['/data/preprocessed_emg_new/' disubj(subj).name '_block' num2str(block_) '_cond' num2str(cond) '.csv'])

            end
        end
    end
end
