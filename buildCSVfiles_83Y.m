warning('off')
clear all
disubj = dir('/data/data_83Y/83Y*');  % Set this to the data path
load('/data/data_83Y/Cond_pourMartin.mat') % Load condition matrix

for subj = 1:length(disubj)
    di = dir(['/data/data_83Y/' disubj(subj).name filesep 'rml_FTT_*.mat']);
    for file = 1:length(di)
        load(['/data/data_83Y/' disubj(subj).name filesep di(file).name]) % Set this to the data path
        if length(disubj(subj).name) == 5
            participant = str2num(disubj(subj).name(end-1:end));
        else
            participant = str2num(disubj(subj).name(end));
        end
        clear colorFB
        if size(CURSOR,2)>6 && block ~= 0
            cond = Cond_block_matrix_allsubjects(1, block, participant);
            if any(reshape(all(Color_target_alltrials==[.8302 .5 1],2),[size(Color_target_alltrials,1)*size(Color_target_alltrials,3) 1]))
                colorKO = [.8302 .5 1];
                colorOK = [.5 .9495 1];
                % reinforced(file,subj) = false;
                reinforced = false;
            elseif any(reshape(all(Color_target_alltrials==[1 .5 .5],2),[size(Color_target_alltrials,1)*size(Color_target_alltrials,3) 1]))
                colorKO = [1 .5 .5];
                colorOK = [.5 1 .5];
                % reinforced(file,subj) = true;
                reinforced = false;
            else
                colorOK = [.6 .6 .6];
                colorKO = [.6 .6 .6];
                % reinforced(file,subj) = NaN;
                reinforced = NaN;
            end
            if ~isnan(reinforced)
                colorFB = squeeze(all(Color_target_alltrials==colorOK,2));
                T=table(CURSOR(1:420,:),colorFB(1:420,:),repmat(Seq_target(1:420),1,36),'VariableNames',{'cursor','colour','target'});
                writetable(T,['/data/preprocessed_data_83Y/' disubj(subj).name '_block' num2str(block) '_cond' num2str(cond) '.csv'])
            end
        end
    end
end

